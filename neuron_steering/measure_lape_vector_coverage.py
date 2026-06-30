#!/usr/bin/env python3
"""
Measure how much dialect steering-vector energy is covered by LAPE neurons.

For each dialect d, this script computes:

    coverage(d) = ||M_d v_d||_2^2 / ||v_d||_2^2

where v_d is a full layer-by-dimension dialect vector and M_d is the binary
mask induced by LAPE-selected (layer, neuron) pairs for the same dialect.

Important: this comparison is only valid when the vector dimensions live in the
same space as the LAPE neurons. LAPE here selects MLP intermediate neurons, so a
residual-stream vector of shape [layers, hidden_size] is not directly comparable
to LAPE masks of shape [layers, intermediate_size]. In strict mode, such shape
mismatches are reported and cause a nonzero exit after outputs are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_DIALECT_MAP = {
    "MSA": "MSA",
    "CAI": "Cairo",
    "RAB": "Rabat",
    "BEI": "Beirut",
    "DOH": "Doha",
    "ALE": "Aleppo",
    "DAM": "Damascus",
    "JED": "Jeddah",
    "RIY": "Riyadh",
    "TUN": "Tunis",
    "KHA": "Khartoum",
}


SummaryRow = Dict[str, Any]
LayerRow = Dict[str, Any]


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def import_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "This script requires torch to load .pt vector files. "
            "Run it in the same environment used for steering/extraction."
        ) from exc
    return torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure LAPE mask coverage of full dialect vector energy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--neurons_dir",
        required=True,
        help="LAPE output directory containing selected_neurons.csv and run_summary.json.",
    )
    parser.add_argument(
        "--vectors_dir",
        required=True,
        help="Directory containing dialect vector .pt files.",
    )
    parser.add_argument("--out_dir", required=True, help="Directory for coverage outputs.")
    parser.add_argument(
        "--dialects",
        nargs="+",
        default=None,
        help="Dialect labels to evaluate, e.g. CAI RAB. Default: all labels in selected_neurons.csv.",
    )
    parser.add_argument(
        "--dialect_map",
        default="",
        help=(
            "Comma-separated LAPE-to-vector name mapping, e.g. CAI=Cairo,RAB=Rabat. "
            "Entries override built-in mappings."
        ),
    )
    parser.add_argument(
        "--vector_pattern",
        default="{name}_response_avg_diff.pt",
        help="Vector filename pattern. Supports {name} for mapped name and {dialect} for LAPE label.",
    )
    parser.add_argument(
        "--selected_csv",
        default=None,
        help="Selected-neuron CSV path. Default: <neurons_dir>/selected_neurons.csv.",
    )
    parser.add_argument(
        "--layer_alignment",
        choices=["auto", "same", "hidden_states"],
        default="auto",
        help=(
            "How LAPE layer IDs align to vector layer rows. same uses vector[layer]; "
            "hidden_states uses vector[layer + 1] because hidden_states[0] is embeddings; "
            "auto chooses same for equal layer counts and hidden_states for num_layers+1 vectors."
        ),
    )
    parser.add_argument(
        "--layer_offset",
        type=int,
        default=None,
        help="Manual vector layer offset. Overrides --layer_alignment.",
    )
    parser.add_argument(
        "--random_baseline",
        type=int,
        default=0,
        help="Number of same-size random masks to sample per dialect. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for baseline masks.")
    parser.add_argument(
        "--allow_unsafe_pickle",
        action="store_true",
        help="Allow torch.load without weights_only=True for trusted local artifacts.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero if requested vectors are incompatible with the LAPE mask.",
    )
    return parser.parse_args()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_dialect_map(text: str) -> Dict[str, str]:
    mapping = dict(DEFAULT_DIALECT_MAP)
    if not text.strip():
        return mapping
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --dialect_map entry {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid --dialect_map entry {item!r}; expected KEY=VALUE.")
        mapping[key] = value
    return mapping


def infer_dialect_order(neurons_dir: Path) -> List[str]:
    run_summary = read_json(neurons_dir / "run_summary.json", {}) or {}
    if isinstance(run_summary.get("dialects"), list):
        return [str(x) for x in run_summary["dialects"]]

    dialects_json = read_json(neurons_dir / "dialects.json", None)
    if isinstance(dialects_json, list):
        return [str(x) for x in dialects_json]
    if isinstance(dialects_json, dict):
        if all(str(k).isdigit() for k in dialects_json):
            return [str(v) for _, v in sorted((int(k), v) for k, v in dialects_json.items())]
        if all(str(v).isdigit() for v in dialects_json.values()):
            return [str(k) for k, _ in sorted(dialects_json.items(), key=lambda kv: int(kv[1]))]
    return []


def load_lape_shape(neurons_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    run_summary = read_json(neurons_dir / "run_summary.json", {}) or {}
    num_layers = run_summary.get("num_layers")
    intermediate_size = run_summary.get("intermediate_size")
    try:
        num_layers = int(num_layers) if num_layers is not None else None
    except (TypeError, ValueError):
        num_layers = None
    try:
        intermediate_size = int(intermediate_size) if intermediate_size is not None else None
    except (TypeError, ValueError):
        intermediate_size = None
    return num_layers, intermediate_size


def load_selected_neurons(
    selected_csv: Path,
    dialect_order: Sequence[str],
) -> Dict[str, Dict[int, Set[int]]]:
    if not selected_csv.exists():
        raise FileNotFoundError(f"selected_neurons.csv not found: {selected_csv}")

    selected: Dict[str, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    with selected_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dialect", "layer", "neuron"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{selected_csv} is missing required columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            dialect = str(row.get("dialect", "")).strip()
            try:
                layer = int(float(str(row.get("layer", "")).strip()))
                neuron = int(float(str(row.get("neuron", "")).strip()))
            except ValueError as exc:
                raise ValueError(f"Invalid layer/neuron at {selected_csv}:{line_no}: {row}") from exc
            if dialect:
                selected[dialect][layer].add(neuron)

    for dialect in dialect_order:
        selected.setdefault(dialect, defaultdict(set))
    return selected


def count_selected(d: Dict[int, Set[int]]) -> int:
    return sum(len(v) for v in d.values())


def format_shape(obj: Any) -> str:
    shape = getattr(obj, "shape", None)
    if shape is None:
        return ""
    return "[" + ", ".join(str(int(x)) for x in shape) + "]"


def torch_load(torch: Any, path: Path, allow_unsafe_pickle: bool = False) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        if allow_unsafe_pickle:
            return torch.load(path, map_location="cpu")
        raise RuntimeError(
            f"Could not safely load {path}. If this is a trusted local artifact, "
            "retry with --allow_unsafe_pickle. Original error: " + repr(exc)
        ) from exc


def normalize_vector_object(torch: Any, obj: Any, source: Path) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        for key in ["response_avg_diff", "vector", "vectors", "steering_vector"]:
            value = obj.get(key)
            if torch.is_tensor(value):
                return value.detach().cpu()
        raise ValueError(
            f"{source} loaded as a dict, but no known tensor key was found. "
            "Tried: response_avg_diff, vector, vectors, steering_vector."
        )

    if isinstance(obj, (list, tuple)) and obj and all(torch.is_tensor(x) for x in obj):
        return torch.stack([x.detach().cpu() for x in obj], dim=0)

    raise ValueError(f"{source} did not load as a tensor, tensor dict, or list of tensors.")


def find_vector_file(
    vectors_dir: Path,
    dialect: str,
    dialect_map: Dict[str, str],
    vector_pattern: str,
) -> Tuple[Optional[Path], str, List[Path]]:
    mapped_name = dialect_map.get(dialect, dialect)
    candidates: List[Path] = []
    for name in [mapped_name, dialect]:
        filename = vector_pattern.format(name=name, dialect=dialect)
        candidates.append(vectors_dir / filename)
        if not filename.endswith(".pt"):
            candidates.append(vectors_dir / f"{filename}.pt")

    seen: Set[Path] = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    for path in unique_candidates:
        if path.exists():
            return path, mapped_name, unique_candidates
    return None, mapped_name, unique_candidates


def infer_layer_offset(
    vector_layers: int,
    lape_layers: int,
    layer_alignment: str,
    manual_offset: Optional[int],
) -> Tuple[Optional[int], Optional[str]]:
    if manual_offset is not None:
        if manual_offset < 0:
            return None, "--layer_offset must be non-negative."
        if vector_layers < lape_layers + manual_offset:
            return (
                None,
                f"Vector has {vector_layers} layers, but LAPE needs {lape_layers} "
                f"layers with offset {manual_offset}.",
            )
        return manual_offset, "manual"

    if layer_alignment == "same":
        if vector_layers < lape_layers:
            return None, f"Vector has {vector_layers} layers, fewer than LAPE layers {lape_layers}."
        return 0, "same"

    if layer_alignment == "hidden_states":
        if vector_layers < lape_layers + 1:
            return None, (
                f"Vector has {vector_layers} layers, but hidden_states alignment needs "
                f"at least {lape_layers + 1}."
            )
        return 1, "hidden_states"

    if vector_layers == lape_layers:
        return 0, "auto:same"
    if vector_layers == lape_layers + 1:
        return 1, "auto:hidden_states"
    return (
        None,
        f"Could not infer layer alignment: vector has {vector_layers} layers, "
        f"LAPE has {lape_layers}. Use --layer_offset to override.",
    )


def empty_summary_row(
    dialect: str,
    vector_name: str,
    vector_file: Optional[Path],
    status: str,
    reason: str,
    selected_neurons: int,
) -> SummaryRow:
    return {
        "status": status,
        "reason": reason,
        "dialect": dialect,
        "vector_name": vector_name,
        "vector_file": str(vector_file) if vector_file is not None else "",
        "vector_shape": "",
        "layer_alignment": "",
        "layer_offset": "",
        "lape_layers": "",
        "vector_layers": "",
        "lape_dim": "",
        "vector_dim": "",
        "selected_neurons": selected_neurons,
        "total_dimensions": "",
        "selected_dimension_fraction": "",
        "total_energy": "",
        "selected_energy": "",
        "complement_energy": "",
        "coverage": "",
        "coverage_percent": "",
    }


def compute_random_baseline(
    torch: Any,
    energy_by_layer: Sequence[Any],
    selected_by_layer: Dict[int, Set[int]],
    dim: int,
    n_samples: int,
    seed: int,
) -> Dict[str, Any]:
    if n_samples <= 0:
        return {}

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    samples: List[float] = []
    layer_counts = {layer: len(neurons) for layer, neurons in selected_by_layer.items() if len(neurons) > 0}

    for _ in range(n_samples):
        energy = 0.0
        for layer, count in layer_counts.items():
            if count <= 0:
                continue
            indices = torch.randperm(dim, generator=generator)[:count]
            energy += float(energy_by_layer[layer].index_select(0, indices).sum().item())
        samples.append(energy)

    if not samples:
        return {
            "random_samples": n_samples,
            "random_selected_energy_mean": 0.0,
            "random_selected_energy_std": 0.0,
            "random_coverage_mean": 0.0,
            "random_coverage_std": 0.0,
        }

    ordered = sorted(samples)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = p * (len(ordered) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return ordered[lo]
        frac = idx - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    mean = statistics.fmean(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return {
        "random_samples": n_samples,
        "random_selected_energy_mean": mean,
        "random_selected_energy_std": std,
        "random_selected_energy_p05": percentile(0.05),
        "random_selected_energy_p50": percentile(0.50),
        "random_selected_energy_p95": percentile(0.95),
    }


def measure_one(
    torch: Any,
    dialect: str,
    vector_name: str,
    vector_file: Path,
    selected_by_layer: Dict[int, Set[int]],
    lape_layers: int,
    lape_dim: int,
    args: argparse.Namespace,
) -> Tuple[SummaryRow, List[LayerRow], bool]:
    selected_neurons = count_selected(selected_by_layer)
    obj = torch_load(torch, vector_file, args.allow_unsafe_pickle)
    vector = normalize_vector_object(torch, obj, vector_file).to(dtype=torch.float64)

    row_base: SummaryRow = {
        "dialect": dialect,
        "vector_name": vector_name,
        "vector_file": str(vector_file),
        "vector_shape": format_shape(vector),
        "lape_layers": lape_layers,
        "lape_dim": lape_dim,
        "selected_neurons": selected_neurons,
    }

    if vector.ndim != 2:
        row = dict(row_base)
        row.update(
            {
                "status": "bad_vector_rank",
                "reason": f"Expected a 2-D [layers, dimensions] tensor, got rank {int(vector.ndim)}.",
            }
        )
        return row, [], True

    vector_layers = int(vector.shape[0])
    vector_dim = int(vector.shape[1])
    offset, alignment_note = infer_layer_offset(
        vector_layers=vector_layers,
        lape_layers=lape_layers,
        layer_alignment=args.layer_alignment,
        manual_offset=args.layer_offset,
    )
    if offset is None:
        row = dict(row_base)
        row.update(
            {
                "status": "layer_mismatch",
                "reason": alignment_note,
                "vector_layers": vector_layers,
                "vector_dim": vector_dim,
            }
        )
        return row, [], True

    if vector_dim != lape_dim:
        row = dict(row_base)
        row.update(
            {
                "status": "dimension_mismatch",
                "reason": (
                    f"Vector dimension is {vector_dim}, but LAPE selected neurons expect "
                    f"intermediate size {lape_dim}. This usually means the vector is in "
                    "residual hidden-state space while LAPE neurons are MLP-intermediate units."
                ),
                "layer_alignment": alignment_note,
                "layer_offset": offset,
                "vector_layers": vector_layers,
                "vector_dim": vector_dim,
            }
        )
        return row, [], True

    invalid: List[Tuple[int, int]] = []
    for layer, neurons in selected_by_layer.items():
        if layer < 0 or layer >= lape_layers:
            invalid.extend((layer, n) for n in sorted(neurons))
        else:
            invalid.extend((layer, n) for n in sorted(neurons) if n < 0 or n >= vector_dim)

    if invalid:
        examples = ", ".join(f"({l},{n})" for l, n in invalid[:8])
        row = dict(row_base)
        row.update(
            {
                "status": "invalid_selected_index",
                "reason": f"{len(invalid)} selected indices are outside vector bounds. Examples: {examples}",
                "layer_alignment": alignment_note,
                "layer_offset": offset,
                "vector_layers": vector_layers,
                "vector_dim": vector_dim,
            }
        )
        return row, [], True

    total_energy = 0.0
    selected_energy = 0.0
    layer_rows: List[LayerRow] = []
    energy_by_lape_layer: List[Any] = []

    for layer in range(lape_layers):
        vector_layer = layer + offset
        energy = vector[vector_layer].pow(2)
        energy_by_lape_layer.append(energy)
        layer_total = float(energy.sum().item())
        neurons = sorted(selected_by_layer.get(layer, set()))
        if neurons:
            idx = torch.tensor(neurons, dtype=torch.long)
            layer_selected = float(energy.index_select(0, idx).sum().item())
        else:
            layer_selected = 0.0
        layer_coverage = layer_selected / layer_total if layer_total > 0 else 0.0
        total_energy += layer_total
        selected_energy += layer_selected
        layer_rows.append(
            {
                "dialect": dialect,
                "layer": layer,
                "vector_layer": vector_layer,
                "layer_total_energy": layer_total,
                "layer_selected_energy": layer_selected,
                "layer_complement_energy": layer_total - layer_selected,
                "layer_coverage": layer_coverage,
                "layer_coverage_percent": 100.0 * layer_coverage,
                "selected_neurons_in_layer": len(neurons),
                "layer_dimensions": vector_dim,
                "selected_dimension_fraction_in_layer": len(neurons) / vector_dim if vector_dim > 0 else 0.0,
            }
        )

    coverage = selected_energy / total_energy if total_energy > 0 else 0.0
    total_dimensions = lape_layers * vector_dim
    summary = dict(row_base)
    summary.update(
        {
            "status": "ok",
            "reason": "",
            "layer_alignment": alignment_note,
            "layer_offset": offset,
            "vector_layers": vector_layers,
            "vector_dim": vector_dim,
            "total_dimensions": total_dimensions,
            "selected_dimension_fraction": selected_neurons / total_dimensions if total_dimensions > 0 else 0.0,
            "total_energy": total_energy,
            "selected_energy": selected_energy,
            "complement_energy": total_energy - selected_energy,
            "coverage": coverage,
            "coverage_percent": 100.0 * coverage,
        }
    )

    baseline = compute_random_baseline(
        torch=torch,
        energy_by_layer=energy_by_lape_layer,
        selected_by_layer=selected_by_layer,
        dim=vector_dim,
        n_samples=args.random_baseline,
        seed=args.seed,
    )
    if baseline:
        random_mean = baseline["random_selected_energy_mean"]
        random_std = baseline["random_selected_energy_std"]
        summary.update(baseline)
        summary["random_coverage_mean"] = random_mean / total_energy if total_energy > 0 else 0.0
        summary["random_coverage_std"] = random_std / total_energy if total_energy > 0 else 0.0
        summary["random_coverage_p05"] = baseline["random_selected_energy_p05"] / total_energy if total_energy > 0 else 0.0
        summary["random_coverage_p50"] = baseline["random_selected_energy_p50"] / total_energy if total_energy > 0 else 0.0
        summary["random_coverage_p95"] = baseline["random_selected_energy_p95"] / total_energy if total_energy > 0 else 0.0
        summary["coverage_minus_random_mean"] = coverage - summary["random_coverage_mean"]
        if summary["random_coverage_std"] > 0:
            summary["coverage_random_z"] = (coverage - summary["random_coverage_mean"]) / summary["random_coverage_std"]
        else:
            summary["coverage_random_z"] = ""

    return summary, layer_rows, False


def summary_fieldnames(include_random: bool) -> List[str]:
    fields = [
        "status",
        "reason",
        "dialect",
        "vector_name",
        "vector_file",
        "vector_shape",
        "layer_alignment",
        "layer_offset",
        "lape_layers",
        "vector_layers",
        "lape_dim",
        "vector_dim",
        "selected_neurons",
        "total_dimensions",
        "selected_dimension_fraction",
        "total_energy",
        "selected_energy",
        "complement_energy",
        "coverage",
        "coverage_percent",
    ]
    if include_random:
        fields.extend(
            [
                "random_samples",
                "random_selected_energy_mean",
                "random_selected_energy_std",
                "random_selected_energy_p05",
                "random_selected_energy_p50",
                "random_selected_energy_p95",
                "random_coverage_mean",
                "random_coverage_std",
                "random_coverage_p05",
                "random_coverage_p50",
                "random_coverage_p95",
                "coverage_minus_random_mean",
                "coverage_random_z",
            ]
        )
    return fields


def layer_fieldnames() -> List[str]:
    return [
        "dialect",
        "layer",
        "vector_layer",
        "layer_total_energy",
        "layer_selected_energy",
        "layer_complement_energy",
        "layer_coverage",
        "layer_coverage_percent",
        "selected_neurons_in_layer",
        "layer_dimensions",
        "selected_dimension_fraction_in_layer",
    ]


def write_report(
    path: Path,
    args: argparse.Namespace,
    summary_rows: Sequence[SummaryRow],
    lape_layers: Optional[int],
    lape_dim: Optional[int],
) -> None:
    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    skipped_rows = [r for r in summary_rows if r.get("status") != "ok"]

    lines = [
        "# LAPE vector-energy coverage",
        "",
        f"Neurons dir: `{Path(args.neurons_dir).resolve()}`",
        f"Vectors dir: `{Path(args.vectors_dir).resolve()}`",
        f"LAPE shape: `{lape_layers} x {lape_dim}`",
        f"Layer alignment: `{args.layer_alignment}`",
        "",
    ]

    if ok_rows:
        lines.extend(
            [
                "## Coverage",
                "",
                "| Dialect | Coverage | Selected energy | Total energy | Selected neurons | Selected dimension share |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in ok_rows:
            lines.append(
                "| {dialect} | {coverage:.4f}% | {selected:.6g} | {total:.6g} | {n} | {frac:.4f}% |".format(
                    dialect=row["dialect"],
                    coverage=float(row["coverage_percent"]),
                    selected=float(row["selected_energy"]),
                    total=float(row["total_energy"]),
                    n=int(row["selected_neurons"]),
                    frac=100.0 * float(row["selected_dimension_fraction"]),
                )
            )
        lines.append("")

        if any("random_coverage_mean" in r for r in ok_rows):
            lines.extend(
                [
                    "## Random Baseline",
                    "",
                    "| Dialect | LAPE coverage | Random mean | Random p05-p95 | Z |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in ok_rows:
                if "random_coverage_mean" not in row:
                    continue
                z = row.get("coverage_random_z", "")
                z_text = "" if z == "" else f"{float(z):.3f}"
                lines.append(
                    "| {dialect} | {coverage:.4f}% | {mean:.4f}% | {p05:.4f}% - {p95:.4f}% | {z} |".format(
                        dialect=row["dialect"],
                        coverage=100.0 * float(row["coverage"]),
                        mean=100.0 * float(row["random_coverage_mean"]),
                        p05=100.0 * float(row["random_coverage_p05"]),
                        p95=100.0 * float(row["random_coverage_p95"]),
                        z=z_text,
                    )
                )
            lines.append("")

        lines.extend(
            [
                "## Interpretation",
                "",
                "Coverage is the fraction of full vector squared L2 energy that lies on LAPE-selected dimensions.",
                "Small values support the claim that the dialect direction is distributed across many dimensions rather than concentrated in the selected neurons.",
                "",
            ]
        )

    if skipped_rows:
        lines.extend(
            [
                "## Not Computed",
                "",
                "| Dialect | Status | Reason |",
                "|---|---|---|",
            ]
        )
        for row in skipped_rows:
            lines.append(f"| {row.get('dialect', '')} | {row.get('status', '')} | {row.get('reason', '')} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch = import_torch()

    neurons_dir = Path(args.neurons_dir).expanduser().resolve()
    vectors_dir = Path(args.vectors_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    selected_csv = Path(args.selected_csv).expanduser().resolve() if args.selected_csv else neurons_dir / "selected_neurons.csv"

    if not neurons_dir.exists():
        raise FileNotFoundError(f"--neurons_dir does not exist: {neurons_dir}")
    if not vectors_dir.exists():
        raise FileNotFoundError(f"--vectors_dir does not exist: {vectors_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    dialect_map = parse_dialect_map(args.dialect_map)
    dialect_order = infer_dialect_order(neurons_dir)
    selected = load_selected_neurons(selected_csv, dialect_order)
    lape_layers, lape_dim = load_lape_shape(neurons_dir)
    if lape_layers is None or lape_dim is None:
        raise ValueError(
            f"Could not infer LAPE num_layers/intermediate_size from {neurons_dir / 'run_summary.json'}."
        )

    if args.dialects:
        dialects = [str(d) for d in args.dialects]
    elif dialect_order:
        dialects = [d for d in dialect_order if d in selected]
    else:
        dialects = sorted(selected)

    summary_rows: List[SummaryRow] = []
    layer_rows: List[LayerRow] = []
    fatal_errors = False

    eprint(f"LAPE mask shape: {lape_layers} layers x {lape_dim} dimensions")
    eprint(f"Evaluating dialects: {', '.join(dialects)}")

    for dialect in dialects:
        selected_by_layer = selected.get(dialect, {})
        selected_count = count_selected(selected_by_layer)
        vector_file, vector_name, candidates = find_vector_file(
            vectors_dir=vectors_dir,
            dialect=dialect,
            dialect_map=dialect_map,
            vector_pattern=args.vector_pattern,
        )
        if vector_file is None:
            reason = "No vector file found. Tried: " + ", ".join(str(p) for p in candidates)
            summary_rows.append(
                empty_summary_row(
                    dialect=dialect,
                    vector_name=vector_name,
                    vector_file=None,
                    status="missing_vector",
                    reason=reason,
                    selected_neurons=selected_count,
                )
            )
            if args.dialects:
                fatal_errors = True
            eprint(f"{dialect}: missing vector; skipped")
            continue

        eprint(f"{dialect}: loading {vector_file}")
        row, rows_by_layer, fatal = measure_one(
            torch=torch,
            dialect=dialect,
            vector_name=vector_name,
            vector_file=vector_file,
            selected_by_layer=selected_by_layer,
            lape_layers=lape_layers,
            lape_dim=lape_dim,
            args=args,
        )
        summary_rows.append(row)
        layer_rows.extend(rows_by_layer)
        fatal_errors = fatal_errors or fatal
        if row.get("status") == "ok":
            eprint(f"{dialect}: coverage={float(row['coverage_percent']):.4f}%")
        else:
            eprint(f"{dialect}: {row.get('status')} - {row.get('reason')}")

    include_random = args.random_baseline > 0
    write_csv(out_dir / "coverage_summary.csv", summary_rows, summary_fieldnames(include_random))
    write_csv(out_dir / "coverage_by_layer.csv", layer_rows, layer_fieldnames())
    write_json(
        out_dir / "coverage_summary.json",
        {
            "neurons_dir": str(neurons_dir),
            "vectors_dir": str(vectors_dir),
            "selected_csv": str(selected_csv),
            "out_dir": str(out_dir),
            "lape_layers": lape_layers,
            "lape_dim": lape_dim,
            "dialects": dialects,
            "dialect_map": dialect_map,
            "vector_pattern": args.vector_pattern,
            "layer_alignment": args.layer_alignment,
            "layer_offset": args.layer_offset,
            "strict": args.strict,
            "random_baseline": args.random_baseline,
            "summary": summary_rows,
        },
    )
    write_report(out_dir / "REPORT.md", args, summary_rows, lape_layers, lape_dim)

    eprint(f"Wrote: {out_dir / 'coverage_summary.csv'}")
    eprint(f"Wrote: {out_dir / 'coverage_by_layer.csv'}")
    eprint(f"Wrote: {out_dir / 'coverage_summary.json'}")
    eprint(f"Wrote: {out_dir / 'REPORT.md'}")

    if args.strict and fatal_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
