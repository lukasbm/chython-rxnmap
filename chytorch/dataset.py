from __future__ import annotations

import errno
import atexit
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback
from io import StringIO
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return
        raise


def get_adm(mol, max_distance=4):
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm > 100] = -1  # remote (different molecule)
    dm[dm > max_distance] = max_distance + 1  # remote (same molecule)
    dm[dm == -1] = max_distance + 2  # remote (different molecule)
    return dm


def _reaction_mols(rxn: str):
    if not _valid_rxn(rxn):
        return None, None
    r, p = [Chem.MolFromSmiles(smi) for smi in rxn.split(">>")]
    return r, p


@lru_cache(maxsize=100_000)
def product_has_invalid_mapping(rxn: str) -> bool:
    r, p = _reaction_mols(rxn)
    if r is None or p is None:
        return True

    r_map_counts: dict[int, int] = {}
    for atom in r.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            r_map_counts[map_num] = r_map_counts.get(map_num, 0) + 1

    pmaps = [atom.GetAtomMapNum() for atom in p.GetAtoms()]
    return (
        0 in pmaps
        or len(pmaps) != len(set(pmaps))
        or any(r_map_counts.get(map_num, 0) != 1 for map_num in pmaps)
    )


def product_is_unmapped(rxn):
    return product_has_invalid_mapping(rxn)


def _mapped_product_maps(p: Chem.Mol) -> set[int]:
    return {atom.GetAtomMapNum() for atom in p.GetAtoms() if atom.GetAtomMapNum() > 0}


def _morgan_fp(mol: Chem.Mol):
    try:
        Chem.GetSymmSSSR(mol)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        try:
            mol = Chem.Mol(mol)
            Chem.SanitizeMol(mol)
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except Exception:
            return None


@lru_cache(maxsize=100_000)
def has_confusing_reagent(rxn: str, tanimoto_threshold: float = 0.5) -> bool:
    r, p = _reaction_mols(rxn)
    if r is None or p is None:
        return True

    product_maps = _mapped_product_maps(p)
    product_fp = _morgan_fp(p)
    if product_fp is None:
        return True

    try:
        reactants = Chem.GetMolFrags(r, asMols=True, sanitizeFrags=True)
    except Exception:
        reactants = Chem.GetMolFrags(r, asMols=True, sanitizeFrags=False)
    for reactant in reactants:
        reactant_maps = {
            atom.GetAtomMapNum()
            for atom in reactant.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }
        if reactant_maps & product_maps:
            continue
        reactant_fp = _morgan_fp(reactant)
        if reactant_fp is None:
            continue
        if DataStructs.TanimotoSimilarity(reactant_fp, product_fp) >= tanimoto_threshold:
            return True
    return False


@lru_cache(maxsize=100_000)
def reaction_passes_paper_filters(rxn: str) -> bool:
    if product_has_invalid_mapping(rxn):
        return False
    if os.environ.get("LOCALMAPPER_CONFUSING_REAGENT_FILTER", "0") != "1":
        return True
    return not has_confusing_reagent(rxn)


def get_mapping_label(rxn):
    rsmi, psmi = rxn.split(">>")
    rmol = Chem.MolFromSmiles(rsmi)
    pmol = Chem.MolFromSmiles(psmi)
    if rmol is None or pmol is None or product_has_invalid_mapping(rxn):
        return None
    r_atom_dict = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in rmol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    labels = []
    for atom in pmol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num not in r_atom_dict:
            return None
        labels.append(r_atom_dict[map_num])
    return labels


def clean_reactant_map(rxn):
    r, p = rxn.split(">>")
    r_mol = Chem.MolFromSmiles(r)
    [atom.SetAtomMapNum(0) for atom in r_mol.GetAtoms()]
    r = Chem.MolToSmiles(r_mol, canonical=False)
    return ">>".join([r, p])


def canonicalize_map_rxn(rxn):
    new_rxn = []
    for smi in rxn.split(">>"):
        mol = Chem.MolFromSmiles(smi)
        index2mapnums = {}
        for atom in mol.GetAtoms():
            index2mapnums[atom.GetIdx()] = atom.GetAtomMapNum()
        mol_cano = Chem.RWMol(mol)
        [atom.SetAtomMapNum(0) for atom in mol_cano.GetAtoms()]
        smi_cano = Chem.MolToSmiles(mol_cano)
        mol_cano = Chem.MolFromSmiles(smi_cano)
        match = mol.GetSubstructMatch(mol_cano)
        if match:
            for atom, mat in zip(mol_cano.GetAtoms(), match):
                atom.SetAtomMapNum(index2mapnums[mat])
            smi = Chem.MolToSmiles(mol_cano, canonical=False)
        new_rxn.append(smi)
    return ">>".join(new_rxn)


def _load_cgr_smiles_reader():
    try:
        from CGRtools import SMILESRead

        return SMILESRead
    except Exception:
        try:
            from CGRtools.files import SMILESRead

            return SMILESRead
        except Exception:
            return None


def cgrtools_available() -> bool:
    return _load_cgr_smiles_reader() is not None


def mapping_comparison_backend() -> str:
    return "cgrtools"


def _cgr_signature_direct(rxn: str) -> str | None:
    if get_mapping_label(rxn) is None:
        return None

    smiles_read = _load_cgr_smiles_reader()
    if smiles_read is None:
        return None

    with smiles_read(
        StringIO(f"{rxn}\n"),
        ignore=_cgrtools_ignore_parser_errors(),
        store_log=bool(_cgr_debug_path()),
    ) as reader:
        reaction = next(iter(reader))
        if _cgr_debug_path():
            parser_log = reaction.meta.get("CGRtoolsParserLog", "")
            if parser_log:
                _log_cgr_event("cgrtools_parser_log", rxn, detail=parser_log.replace("\n", " | "))
    cgr = reaction.compose()
    return str(cgr)


@lru_cache(maxsize=100_000)
def cgr_signature(rxn: str) -> str | None:
    return _cgr_signature_isolated(rxn)


def mapping_signature(rxn: str) -> tuple[tuple[str, str], tuple[tuple[int, int], ...]] | None:
    try:
        rxn = canonicalize_map_rxn(rxn)
        rsmi, psmi = rxn.split(">>")
        rmol = Chem.MolFromSmiles(rsmi)
        pmol = Chem.MolFromSmiles(psmi)
    except Exception:
        return None
    if rmol is None or pmol is None:
        return None

    r_demapped = Chem.RWMol(rmol)
    p_demapped = Chem.RWMol(pmol)
    for mol in (r_demapped, p_demapped):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    pattern = (
        Chem.MolToSmiles(r_demapped, canonical=True, isomericSmiles=True),
        Chem.MolToSmiles(p_demapped, canonical=True, isomericSmiles=True),
    )

    reactant_map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in rmol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    pairs = []
    for product_idx, atom in enumerate(pmol.GetAtoms()):
        map_num = atom.GetAtomMapNum()
        if map_num <= 0 or map_num not in reactant_map_to_idx:
            return None
        pairs.append((product_idx, reactant_map_to_idx[map_num]))
    return pattern, tuple(pairs)


def mappings_are_equivalent(predicted_rxn: str, reference_rxn: str) -> bool:
    predicted_cgr = cgr_signature(predicted_rxn)
    reference_cgr = cgr_signature(reference_rxn)
    return predicted_cgr is not None and predicted_cgr == reference_cgr


def mapping_matches_any(predicted_rxn: str, reference_rxns: list[str]) -> bool:
    return any(mappings_are_equivalent(predicted_rxn, reference) for reference in reference_rxns)


def normalize_mapped_rxn(rxn: str) -> str | None:
    if get_mapping_label(rxn) is None or not reaction_passes_paper_filters(rxn):
        return None
    if os.environ.get("LOCALMAPPER_CANONICALIZE_TARGETS", "0") != "1":
        return rxn
    try:
        normalized = canonicalize_map_rxn(rxn)
    except Exception:
        return None
    return normalized if get_mapping_label(normalized) is not None else None


def _valid_rxn(rxn: Any) -> bool:
    if not isinstance(rxn, str) or rxn.count(">>") != 1:
        return False
    reactants, products = rxn.split(">>")
    return bool(reactants and products)


def _alternatives(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [
        part.strip() for part in value.strip().split(",") if _valid_rxn(part.strip())
    ]


def _item(
    idx: Any,
    rxn: Any,
    *,
    split: str | None = None,
    source: Any = None,
    alternatives: list[str] | None = None,
) -> dict[str, Any] | None:
    rxns = alternatives if alternatives is not None else _alternatives(rxn)
    if not rxns:
        return None
    return {
        "id": str(idx),
        "rxn": rxns[0],
        "mapped_rxns": rxns,
        "split": split,
        "original_split": split,
        "source": None if pd.isna(source) else source,
        "num_mappings": len(rxns),
    }


def _from_frame(
    df: pd.DataFrame,
    rxn_col: str,
    *,
    split: str | None = None,
    id_col: str | None = None,
    source_col: str | None = None,
) -> list[dict[str, Any]]:
    items = []
    for i, row in df.iterrows():
        item = _item(
            row[id_col] if id_col and id_col in df.columns else i,
            row[rxn_col],
            split=split,
            source=row[source_col] if source_col and source_col in df.columns else None,
        )
        if item is not None:
            items.append(item)
    return items


def _from_line_file(path: Path, *, split: str, id_prefix: str | None = None) -> list[dict[str, Any]]:
    items = []
    with path.open() as f:
        for i, line in enumerate(f):
            rxns = _alternatives(line.rstrip("\n\r"))
            idx = f"{id_prefix}:{i}" if id_prefix else i
            item = _item(idx, None, split=split, alternatives=rxns)
            if item is not None:
                items.append(item)
    return items


def _from_semicolon_file(path: Path, *, split: str, id_prefix: str | None = None) -> list[dict[str, Any]]:
    items = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n\r")
            if not line:
                continue
            idx, payload = line.split(";", 1) if ";" in line else (i, line)
            idx = f"{id_prefix}:{idx}" if id_prefix else idx
            item = _item(idx, None, split=split, alternatives=_alternatives(payload))
            if item is not None:
                items.append(item)
    return items


class ReactionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mol_to_graph,
        *,
        data_root: str | Path = "data",
        items: list[dict[str, Any]] | None = None,
        include_labels: bool = True,
    ):
        self.mol_to_graph = mol_to_graph
        self.include_labels = include_labels
        self.data_root = Path(data_root)
        self.items = self.load_items() if items is None else items

    def load_items(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def from_items(
        cls,
        items: list[dict[str, Any]],
        mol_to_graph,
        *,
        include_labels: bool = True,
    ) -> "ReactionDataset":
        return cls(mol_to_graph, items=items, include_labels=include_labels)

    def __getitem__(self, item):
        data = self.items[item]
        rxn = data["rxn"]
        r, p = rxn.split(">>")
        rgraph = self.mol_to_graph(Chem.MolFromSmiles(r))
        pgraph = self.mol_to_graph(Chem.MolFromSmiles(p))
        label = get_mapping_label(rxn) if self.include_labels else []
        if self.include_labels and label is None:
            raise ValueError(f"Invalid mapped reaction for training item: {data['id']}")
        return data["id"], rxn, rgraph, pgraph, label, data.get("weight", 1.0), data

    def __len__(self):
        return len(self.items)


class USPTO50KDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        df = pd.read_csv(self.data_root / "USPTO_50K" / "raw_data.csv")
        return _from_frame(df, "mapped_rxn")


class GoldenDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        df = pd.read_csv(self.data_root / "Golden" / "raw_data.csv")
        return _from_frame(df, "mapped_rxn")


class NatCommDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        df = pd.read_csv(self.data_root / "NatComm" / "test_data.csv")
        return _from_frame(df, "mapped_rxn", source_col="source")


class SchneiderDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        df = pd.read_csv(self.data_root / "schneider" / "schneider50k.tsv", sep="\t")
        return _from_frame(df, "clean_rxn", id_col="Unnamed: 0", source_col="source")


class RingReactionsDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        return _from_line_file(
            self.data_root / "ringreactions" / "train_ringreactions.csv",
            split=None,
            id_prefix="train",
        ) + _from_line_file(
            self.data_root / "ringreactions" / "test_ringreactions.csv",
            split=None,
            id_prefix="test",
        )


class MetAMDBDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        return _from_semicolon_file(
            self.data_root / "metAMDB" / "train_metamdb_filtered.csv",
            split=None,
            id_prefix="train",
        ) + _from_semicolon_file(
            self.data_root / "metAMDB" / "test_metamdb_filtered.csv",
            split=None,
            id_prefix="test",
        )


DATASETS = {
    "uspto_50k": USPTO50KDataset,
    "uspto50k": USPTO50KDataset,
    "golden": GoldenDataset,
    "natcomm": NatCommDataset,
    "jaworski": NatCommDataset,
    "schneider": SchneiderDataset,
    "schneider50k": SchneiderDataset,
    "ringreactions": RingReactionsDataset,
    "ring_reactions": RingReactionsDataset,
    "metamdb": MetAMDBDataset,
    "metamdb_filtered": MetAMDBDataset,
}


def _dataset_key(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def create_reaction_dataset(
    dataset: str,
    mol_to_graph,
    *,
    data_root: str | Path = "data",
    include_labels: bool = True,
) -> ReactionDataset:
    dataset_cls = DATASETS.get(_dataset_key(dataset))
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    dataset_obj = dataset_cls(
        mol_to_graph,
        data_root=data_root,
        include_labels=include_labels,
    )
    return dataset_obj


def normalize_item_target(item: dict[str, Any]) -> dict[str, Any] | None:
    reference_rxns = item.get("mapped_rxns", [item["rxn"]])
    normalized_rxns = [
        normalized
        for reference_rxn in reference_rxns
        if (normalized := normalize_mapped_rxn(reference_rxn))
    ]
    if not normalized_rxns:
        return None
    return {
        **item,
        "rxn": normalized_rxns[0],
        "mapped_rxns": normalized_rxns,
        "num_mappings": len(normalized_rxns),
    }


class FilterTimeoutError(TimeoutError):
    pass


def _handle_filter_timeout(signum, frame):
    raise FilterTimeoutError


def _filter_timeout_seconds() -> float:
    try:
        return float(os.environ.get("LOCALMAPPER_FILTER_TIMEOUT_SECONDS", "5"))
    except ValueError:
        return 5.0


def _cgr_timeout_seconds() -> float:
    try:
        return float(os.environ.get("LOCALMAPPER_CGR_TIMEOUT_SECONDS", "2"))
    except ValueError:
        return 2.0


def _cgr_debug_path() -> Path | None:
    path = os.environ.get("LOCALMAPPER_CGR_DEBUG_PATH")
    return Path(path) if path else None


def _cgrtools_ignore_parser_errors() -> bool:
    value = os.environ.get("LOCALMAPPER_CGRTOOLS_IGNORE", "1").lower()
    return value not in {"0", "false", "off", "no"}


def _log_cgr_event(event: str, rxn: str, *, detail: str = ""):
    path = _cgr_debug_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(f"{time.time():.6f}\t{event}\t{detail}\t{rxn}\n")
    except Exception:
        pass


_CGR_WORKER = None
_CGR_TASK_QUEUE = None
_CGR_RESULT_QUEUE = None
_CGR_TASK_ID = 0


def _cgr_worker_loop(task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_id, rxn = task
        try:
            signature = _cgr_signature_direct(rxn)
            error = ""
        except Exception:
            signature = None
            error = traceback.format_exc(limit=1).strip().replace("\n", " | ")
        result_queue.put((task_id, signature, error))


def _stop_cgr_worker():
    global _CGR_WORKER, _CGR_TASK_QUEUE, _CGR_RESULT_QUEUE
    worker = _CGR_WORKER
    if worker is not None and worker.is_alive():
        try:
            _CGR_TASK_QUEUE.put_nowait(None)
            worker.join(timeout=0.2)
        except Exception:
            pass
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=0.2)
    _CGR_WORKER = None
    _CGR_TASK_QUEUE = None
    _CGR_RESULT_QUEUE = None


def _ensure_cgr_worker():
    global _CGR_WORKER, _CGR_TASK_QUEUE, _CGR_RESULT_QUEUE
    if _CGR_WORKER is not None and _CGR_WORKER.is_alive():
        return
    _stop_cgr_worker()
    context = mp.get_context(os.environ.get("LOCALMAPPER_CGR_MP_CONTEXT", "fork"))
    _CGR_TASK_QUEUE = context.Queue(maxsize=1)
    _CGR_RESULT_QUEUE = context.Queue(maxsize=1)
    _CGR_WORKER = context.Process(
        target=_cgr_worker_loop,
        args=(_CGR_TASK_QUEUE, _CGR_RESULT_QUEUE),
        daemon=True,
    )
    _CGR_WORKER.start()


def _cgr_signature_isolated(rxn: str) -> str | None:
    global _CGR_TASK_ID
    if threading.current_thread() is not threading.main_thread():
        try:
            return _cgr_signature_direct(rxn)
        except Exception:
            return None

    timeout = _cgr_timeout_seconds()
    if timeout <= 0:
        try:
            return _cgr_signature_direct(rxn)
        except Exception:
            return None

    _ensure_cgr_worker()
    _CGR_TASK_ID += 1
    task_id = _CGR_TASK_ID
    try:
        _CGR_TASK_QUEUE.put((task_id, rxn), timeout=timeout)
        while True:
            result_id, signature, error = _CGR_RESULT_QUEUE.get(timeout=timeout)
            if result_id == task_id:
                if signature is None:
                    _log_cgr_event("cgrtools_none", rxn, detail=error)
                return signature
    except (queue.Empty, queue.Full):
        _log_cgr_event("cgrtools_timeout", rxn, detail=f"timeout={timeout}")
        _stop_cgr_worker()
        return None


atexit.register(_stop_cgr_worker)


def normalize_item_target_with_timeout(item: dict[str, Any]) -> dict[str, Any] | None:
    timeout = _filter_timeout_seconds()
    if timeout <= 0 or threading.current_thread() is not threading.main_thread():
        return normalize_item_target(item)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_filter_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return normalize_item_target(item)
    except FilterTimeoutError:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _progress(iterable, *, desc: str, total: int | None = None):
    if tqdm is None or os.environ.get("LOCALMAPPER_PROGRESS", "1") == "0":
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def normalize_item_targets(
    items: list[dict[str, Any]],
    *,
    progress_desc: str | None = None,
) -> list[dict[str, Any]]:
    iterator = (
        _progress(items, desc=progress_desc, total=len(items))
        if progress_desc
        else items
    )
    return [
        normalized
        for item in iterator
        if (normalized := normalize_item_target_with_timeout(item)) is not None
    ]


def item_has_valid_mapping(item: dict[str, Any]) -> bool:
    return any(
        get_mapping_label(reference_rxn) is not None
        and reaction_passes_paper_filters(reference_rxn)
        for reference_rxn in item.get("mapped_rxns", [item["rxn"]])
    )


def select_split(
    items: list[dict[str, Any]],
    split: str,
    *,
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    split = split.lower()
    explicit = [item for item in items if item["split"] == split]
    if explicit:
        return normalize_item_targets(
            explicit,
            progress_desc=f"Filtering {split} mappings",
        )

    unsplit = [item for item in items if item["split"] is None]
    if not unsplit:
        return []

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unsplit))
    n_test = int(round(len(unsplit) * test_fraction))
    n_val = int(round(len(unsplit) * val_fraction))
    test_ids = set(order[:n_test])
    val_ids = set(order[n_test : n_test + n_val])

    selected = []
    for i, item in enumerate(unsplit):
        item_split = "test" if i in test_ids else "val" if i in val_ids else "train"
        if item_split == split:
            selected.append({**item, "split": item_split})
    if max_candidates is not None and len(selected) > max_candidates:
        selected_order = rng.permutation(len(selected))[:max_candidates]
        selected = [selected[i] for i in selected_order]
    return normalize_item_targets(
        selected,
        progress_desc=f"Filtering {split} mappings",
    )


def load_dataset_items(
    dataset: str,
    *,
    data_root: str | Path = "data",
) -> list[dict[str, Any]]:
    # For code that only needs the reaction list, not graph construction.
    return create_reaction_dataset(dataset, lambda mol: mol, data_root=data_root).items


def load_reactions(
    dataset: str,
    split: str,
    *,
    data_root: str | Path = "data",
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    return select_split(
        load_dataset_items(dataset, data_root=data_root),
        split,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        max_candidates=max_candidates,
    )


def collate_reaction_batch(batch):
    batch = [row for row in batch if row is not None]
    idxs, rxns, rgraphs, pgraphs, labels, weights, data = zip(*batch)
    labels_list = [torch.as_tensor(label, dtype=torch.long) for label in labels]
    masks_list = [torch.ones_like(label, dtype=torch.long) for label in labels_list]
    return (
        list(idxs),
        list(rxns),
        list(rgraphs),
        list(pgraphs),
        labels_list,
        masks_list,
        list(weights),
        list(data),
    )
