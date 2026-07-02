from __future__ import annotations

import csv
import json
import platform
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import fire

from nora.datasets import load_reference_rows
from nora.metrics import aggregate_scores, score_prediction, strip_atom_maps


class MappingTimeoutError(Exception):
    pass


@dataclass(frozen=True)
class PredictionResult:
    prediction: str | None
    error: str | None
    elapsed_seconds: float


def write_json(path: Path | str, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(output)


def write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return str(output)


def _safe_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


@contextmanager
def _timeout(seconds: float | None):
    if seconds is None:
        yield
        return

    def handler(_signum: int, _frame: Any) -> None:
        raise MappingTimeoutError(f"mapping timed out after {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _map_with_chython_reset_mapping(unmapped_reaction: str, timeout_seconds: float | None) -> PredictionResult:
    start = time.monotonic()
    try:
        from chython import smiles

        with _timeout(timeout_seconds):
            reaction = smiles(unmapped_reaction)
            reaction.reset_mapping()
        return PredictionResult(
            prediction=format(reaction, "m"),
            error=None,
            elapsed_seconds=time.monotonic() - start,
        )
    except Exception as exc:
        return PredictionResult(
            prediction=None,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - start,
        )


def _map_with_reference_copy(references: list[str], started_at: float) -> PredictionResult:
    return PredictionResult(
        prediction=references[0] if references else None,
        error=None if references else "missing reference",
        elapsed_seconds=time.monotonic() - started_at,
    )


def predict_row(
    *,
    backend: str,
    unmapped_reaction: str,
    references: list[str],
    timeout_seconds: float | None,
) -> PredictionResult:
    start = time.monotonic()
    if backend == "chython_reset_mapping":
        return _map_with_chython_reset_mapping(unmapped_reaction, timeout_seconds)
    if backend == "reference_copy":
        return _map_with_reference_copy(references, start)
    raise ValueError("backend must be 'chython_reset_mapping' or 'reference_copy'")


def run_evaluation(
    *,
    dataset: str,
    split: str,
    data_root: str,
    backend: str,
    mode: str,
    checkpoint: str | None,
    limit: int | None,
    timeout_seconds: float | None,
    output_dir: str,
) -> dict[str, Any]:
    if checkpoint and backend == "chython_reset_mapping":
        raise ValueError(
            "checkpoint was provided, but the chython_reset_mapping backend uses chython's "
            "configured mapper and this repo does not expose a custom-checkpoint injection hook"
        )

    reference_rows = load_reference_rows(dataset=dataset, split=split, root=data_root, limit=limit)
    row_outputs: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []

    for row_index, references in enumerate(reference_rows):
        unmapped_reaction = strip_atom_maps(references[0])
        prediction = predict_row(
            backend=backend,
            unmapped_reaction=unmapped_reaction,
            references=references,
            timeout_seconds=timeout_seconds,
        )
        row_score = score_prediction(prediction.prediction, references, prediction.error)
        scores.append(row_score)
        row_outputs.append(
            {
                "row_index": row_index,
                "dataset": dataset,
                "split": split,
                "mode": mode,
                "backend": backend,
                "unmapped_reaction": unmapped_reaction,
                "references": references,
                "prediction": prediction.prediction,
                "prediction_error": prediction.error,
                "elapsed_seconds": prediction.elapsed_seconds,
                "score": row_score,
            }
        )

    metrics = aggregate_scores(scores)
    output_root = Path(output_dir)
    run_id = f"{dataset}_{split}_{mode}_{backend}"
    predictions_path = output_root / f"predictions_{run_id}.jsonl"
    metrics_path = output_root / f"metrics_{run_id}.json"
    metadata_path = output_root / f"run_metadata_{run_id}.json"

    equivalence_backend = _equivalence_backend_metadata(timeout_seconds)

    metadata_payload = {
        "dataset": dataset,
        "split": split,
        "data_root": data_root,
        "mode": mode,
        "backend": backend,
        "checkpoint": checkpoint,
        "limit": limit,
        "timeout_seconds": timeout_seconds,
        "reference_rows": len(reference_rows),
        "prediction_rows": len(row_outputs),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "chython": _safe_version("chython"),
            "chytorch": _safe_version("chytorch"),
            "pytorch-lightning": _safe_version("pytorch-lightning"),
            "torch": _safe_version("torch"),
        },
        "git": _git_metadata(),
        "equivalence_backend": equivalence_backend,
        "confidence": {
            "status": "not_available",
            "reason": "no scalar confidence extractor has been defined yet",
        },
    }

    result = {
        "metrics": metrics,
        "artifacts": {
            "predictions_jsonl": write_jsonl(predictions_path, row_outputs),
            "metrics_json": write_json(metrics_path, metrics),
            "run_metadata_json": write_json(metadata_path, metadata_payload),
        },
        "metadata": metadata_payload,
    }
    return result


def append_summary_csv(summary_csv: Path | str, result: dict[str, Any]) -> str:
    output = Path(summary_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload = result["metadata"]
    metrics = result["metrics"]
    row = {
        "dataset": metadata_payload["dataset"],
        "split": metadata_payload["split"],
        "mode": metadata_payload["mode"],
        "backend": metadata_payload["backend"],
        "checkpoint": metadata_payload["checkpoint"],
        "total": metrics["aam"]["total"],
        "equiv_exact_match_accuracy": metrics["aam"]["equiv_exact_match_accuracy"],
        "atom_accuracy": metrics["aam"]["atom_accuracy"],
        "raw_exact_match_accuracy": metrics["aam"]["raw_exact_match_accuracy"],
        "invalid_mapping_count": metrics["aam"]["invalid_mapping_count"],
        "binary_exact_f1": metrics["binary_exact"]["f1"],
        "equivalence_backend": metadata_payload["equivalence_backend"]["name"],
    }
    write_header = not output.exists()
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return str(output)


def main(
    dataset: str = "golden",
    split: str = "test",
    data_root: str = "data/data",
    backend: str = "chython_reset_mapping",
    mode: str = "pretrained_zero_shot",
    checkpoint: str | None = None,
    limit: int | None = None,
    timeout_seconds: float | None = 2.0,
    output_dir: str = "experiment_results/evaluation",
    summary_csv: str | None = "experiment_results/evaluation/metrics_summary.csv",
) -> dict[str, Any]:
    result = run_evaluation(
        dataset=dataset,
        split=split,
        data_root=data_root,
        backend=backend,
        mode=mode,
        checkpoint=checkpoint,
        limit=limit,
        timeout_seconds=timeout_seconds,
        output_dir=output_dir,
    )
    if summary_csv:
        result["artifacts"]["summary_csv"] = append_summary_csv(summary_csv, result)
    return result


def _equivalence_backend_metadata(timeout_seconds: float | None) -> dict[str, Any]:
    try:
        from chytorch.dataset import cgrtools_available, mapping_comparison_backend

        if cgrtools_available():
            return {
                "name": mapping_comparison_backend(),
                "parser_policy": "CGRTools SMILESRead with LOCALMAPPER_CGRTOOLS_IGNORE defaulting to 1",
                "timeout_policy": "LOCALMAPPER_CGR_TIMEOUT_SECONDS controls isolated CGR signature timeout",
                "failure_behavior": "failed or timed-out signatures count as non-equivalent",
                "directly_comparable_to_cgrtools": True,
            }
        return {
            "name": "rdkit_mapping_signature",
            "parser_policy": "RDKit canonicalized demapped reaction pattern plus product-to-reactant atom correspondence",
            "timeout_policy": timeout_seconds,
            "failure_behavior": "failed signatures count as non-equivalent",
            "directly_comparable_to_cgrtools": False,
        }
    except Exception:
        pass

    return {
        "name": "map_signature_fallback",
        "parser_policy": "regex atom-map parser over mapped reaction SMILES",
        "timeout_policy": timeout_seconds,
        "failure_behavior": "failed predictions count as invalid and incorrect",
        "directly_comparable_to_cgrtools": False,
    }


if __name__ == "__main__":
    fire.Fire(main)
