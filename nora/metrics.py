from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


ATOM_MAP_RE = re.compile(r":\d+(?=\])")
BRACKET_ATOM_RE = re.compile(r"\[[^\]]+\]")
BRACKET_ATOM_MAP_RE = re.compile(r":(\d+)(?=\])")


@dataclass(frozen=True)
class MappingSignature:
    product_atom_sources: tuple[int, ...]


@dataclass(frozen=True)
class MappingParse:
    signature: MappingSignature | None
    valid: bool
    error: str | None = None


def strip_atom_maps(reaction_smiles: str) -> str:
    return ATOM_MAP_RE.sub("", reaction_smiles)


def split_reaction(reaction_smiles: str) -> tuple[str, str]:
    parts = reaction_smiles.split(">")
    if len(parts) == 3:
        return parts[0], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError("reaction SMILES must contain '>>' or reactants>agents>products")


def _side_map_numbers(side_smiles: str) -> list[int]:
    maps: list[int] = []
    for token in BRACKET_ATOM_RE.findall(side_smiles):
        match = BRACKET_ATOM_MAP_RE.search(token)
        if match:
            maps.append(int(match.group(1)))
    return maps


def parse_mapping_signature(reaction_smiles: str) -> MappingParse:
    try:
        reactants, products = split_reaction(reaction_smiles)
    except ValueError as exc:
        return MappingParse(signature=None, valid=False, error=str(exc))

    reactant_maps = _side_map_numbers(reactants)
    product_maps = _side_map_numbers(products)
    if not reactant_maps:
        return MappingParse(signature=None, valid=False, error="no mapped reactant atoms")
    if not product_maps:
        return MappingParse(signature=None, valid=False, error="no mapped product atoms")
    if len(set(reactant_maps)) != len(reactant_maps):
        return MappingParse(signature=None, valid=False, error="duplicate reactant atom maps")
    if len(set(product_maps)) != len(product_maps):
        return MappingParse(signature=None, valid=False, error="duplicate product atom maps")

    reactant_order = {map_number: index for index, map_number in enumerate(reactant_maps)}
    missing = [map_number for map_number in product_maps if map_number not in reactant_order]
    if missing:
        return MappingParse(
            signature=None,
            valid=False,
            error=f"product maps absent from reactants: {missing[:5]}",
        )

    return MappingParse(
        signature=MappingSignature(tuple(reactant_order[map_number] for map_number in product_maps)),
        valid=True,
    )


def raw_exact_match(prediction: str, references: Iterable[str]) -> bool:
    normalized_prediction = prediction.strip()
    return any(normalized_prediction == reference.strip() for reference in references)


def atom_accuracy(prediction: str, references: Iterable[str]) -> float | None:
    rdkit_score = _rdkit_atom_accuracy(prediction, references)
    if rdkit_score is not None:
        return rdkit_score

    pred_parse = parse_mapping_signature(prediction)
    if pred_parse.signature is None:
        return None

    best: float | None = None
    for reference in references:
        ref_parse = parse_mapping_signature(reference)
        if ref_parse.signature is None:
            continue
        pred = pred_parse.signature.product_atom_sources
        ref = ref_parse.signature.product_atom_sources
        if not ref:
            continue
        matches = sum(1 for left, right in zip(pred, ref) if left == right)
        score = matches / max(len(pred), len(ref))
        best = score if best is None else max(best, score)
    return best


def equiv_exact_match(prediction: str, references: Iterable[str]) -> bool:
    try:
        from chytorch.dataset import cgrtools_available, mapping_matches_any, mapping_signature

        reference_list = list(references)
        if cgrtools_available():
            return mapping_matches_any(prediction, reference_list)

        predicted_signature = mapping_signature(prediction)
        return predicted_signature is not None and any(
            predicted_signature == mapping_signature(reference)
            for reference in reference_list
        )
    except Exception:
        pass

    pred_parse = parse_mapping_signature(prediction)
    if pred_parse.signature is None:
        return False
    return any(
        pred_parse.signature == ref_parse.signature
        for ref_parse in (parse_mapping_signature(reference) for reference in references)
        if ref_parse.signature is not None
    )


def score_prediction(prediction: str | None, references: list[str], error: str | None = None) -> dict[str, Any]:
    if prediction is None:
        return {
            "equiv_exact_match": False,
            "raw_exact_match": False,
            "atom_accuracy": None,
            "valid_mapping": False,
            "error": error or "missing prediction",
        }

    pred_parse = parse_mapping_signature(prediction)
    valid_mapping = _rdkit_valid_mapping(prediction)
    if valid_mapping is None:
        valid_mapping = pred_parse.valid
    return {
        "equiv_exact_match": equiv_exact_match(prediction, references),
        "raw_exact_match": raw_exact_match(prediction, references),
        "atom_accuracy": atom_accuracy(prediction, references),
        "valid_mapping": valid_mapping,
        "error": error or pred_parse.error,
    }


def _rdkit_valid_mapping(prediction: str) -> bool | None:
    try:
        from chytorch.dataset import get_mapping_label
    except Exception:
        return None
    return get_mapping_label(prediction) is not None


def _rdkit_atom_accuracy(prediction: str, references: Iterable[str]) -> float | None:
    try:
        from chytorch.dataset import get_mapping_label
    except Exception:
        return None

    pred_label = get_mapping_label(prediction)
    if pred_label is None:
        return None

    best: float | None = None
    for reference in references:
        ref_label = get_mapping_label(reference)
        if ref_label is None:
            continue
        matches = sum(1 for left, right in zip(pred_label, ref_label) if left == right)
        score = matches / max(len(pred_label), len(ref_label)) if pred_label or ref_label else 0.0
        best = score if best is None else max(best, score)
    return best


def aggregate_scores(row_scores: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(row_scores)
    exact = sum(1 for row in row_scores if row["equiv_exact_match"])
    raw_exact = sum(1 for row in row_scores if row["raw_exact_match"])
    invalid = sum(1 for row in row_scores if not row["valid_mapping"])
    atom_scores = [row["atom_accuracy"] for row in row_scores if row["atom_accuracy"] is not None]

    accuracy = exact / total if total else 0.0
    raw_accuracy = raw_exact / total if total else 0.0
    mean_atom_accuracy = sum(atom_scores) / len(atom_scores) if atom_scores else 0.0

    # With no confidence score, every row is a positive target and the mapper's
    # exact-match decision is the positive prediction. Precision is therefore
    # not a useful discriminator, but the fields keep downstream tables stable.
    recall = accuracy
    precision = 1.0 if exact else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return {
        "aam": {
            "equiv_exact_match_accuracy": accuracy,
            "atom_accuracy": mean_atom_accuracy,
            "raw_exact_match_accuracy": raw_accuracy,
            "invalid_mapping_count": invalid,
            "total": total,
        },
        "binary_exact": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "score_calibration": {
            "status": "not_available",
            "reason": "no comparable scalar confidence score is defined for this mapper backend",
            "ap": None,
            "f1": None,
            "mce": None,
            "ece": None,
        },
    }
