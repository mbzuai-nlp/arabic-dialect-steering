#!/usr/bin/env python3
"""
Analyze LAPE / RUCAIBox-style dialect-neuron extraction outputs.

This script reads an output directory produced by lape_rucai_parallel.py or the
previous lape_extract*.py scripts and creates:
  - CSV tables with selected-neuron counts and score distributions;
  - JSON summary statistics;
  - PNG graphs for layer/dialect distributions and score histograms.

It does NOT rerun the language model. It only analyzes saved artifacts.

Typical usage:
  python analyze_lape_output.py \
    --input_dir outputs/lape_allam \
    --out_dir outputs/lape_allam/analysis

Main expected input files, when available:
  RUCAI-style:
    neurons.pth
    selected_neurons.csv
    activation_probs.pt       # [num_layers, intermediate_size, num_dialects]
    entropy.pt                # [num_layers, intermediate_size]
    entropy_for_selection.pt  # [num_layers, intermediate_size]
    over_zero.pt              # [num_layers, intermediate_size, num_dialects]
    token_counts.pt
    run_summary.json
    thresholds.json
    dialects.json

  Earlier style:
    selected_neurons.csv
    activation_probabilities.pt  # [num_dialects, num_layers, intermediate_size]
    lape_scores.pt               # [num_layers, intermediate_size]
    normalized_lape_scores.pt
    active_counts.pt
    token_counts.pt
    run_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires matplotlib. Install with: pip install matplotlib"
    ) from exc


Number = Optional[float]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create graphs and statistics for LAPE/RUCAIBox dialect-neuron outputs."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing LAPE extraction outputs.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Where to write analysis outputs. Default: <input_dir>/analysis",
    )
    parser.add_argument(
        "--allow_unsafe_pickle",
        action="store_true",
        help=(
            "Allow torch.load fallback without weights_only=True. Only use this for trusted output files. "
            "Most tensor/list-of-tensor artifacts should load without this."
        ),
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=80,
        help="Number of bins for histogram plots.",
    )
    parser.add_argument(
        "--sample_per_layer_boxplot",
        type=int,
        default=4000,
        help=(
            "Maximum number of neurons sampled per layer for entropy boxplot. "
            "Use 0 to include all neurons."
        ),
    )
    parser.add_argument(
        "--top_k_layers",
        type=int,
        default=10,
        help="Number of top layers per dialect to save in top_layers_by_dialect.csv.",
    )
    parser.add_argument(
        "--top_k_neurons",
        type=int,
        default=500,
        help="Number of selected neurons to save in selected_neurons_ranked_topk.csv.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def torch_load(path: Path, allow_unsafe_pickle: bool = False) -> Any:
    """Load a torch artifact as safely as possible."""
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch does not support weights_only.
        if allow_unsafe_pickle:
            return torch.load(path, map_location="cpu")
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            raise
    except Exception as exc:
        if allow_unsafe_pickle:
            return torch.load(path, map_location="cpu")
        raise RuntimeError(
            f"Could not safely load {path}. If this is a trusted file created by your own run, "
            "retry with --allow_unsafe_pickle. Original error: " + repr(exc)
        ) from exc


def find_first_existing(input_dir: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        path = input_dir / name
        if path.exists():
            return path
    return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def infer_dialects(input_dir: Path) -> List[str]:
    run_summary = read_json(input_dir / "run_summary.json", {}) or {}
    if isinstance(run_summary.get("dialects"), list):
        return [str(x) for x in run_summary["dialects"]]

    dialects_json = read_json(input_dir / "dialects.json", None)
    if isinstance(dialects_json, dict):
        # Common format: {"0": "CAI", "1": "MSA", ...}
        try:
            items = sorted(((int(k), str(v)) for k, v in dialects_json.items()), key=lambda x: x[0])
            return [v for _, v in items]
        except Exception:
            return [str(v) for v in dialects_json.values()]
    if isinstance(dialects_json, list):
        return [str(x) for x in dialects_json]

    selected_path = input_dir / "selected_neurons.csv"
    if selected_path.exists():
        dialects = []
        seen = set()
        with selected_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                d = row.get("dialect")
                if d is not None and d not in seen:
                    seen.add(d)
                    dialects.append(d)
        if dialects:
            return dialects

    raise FileNotFoundError(
        "Could not infer dialect labels. Expected run_summary.json, dialects.json, or selected_neurons.csv."
    )


def load_tensor_by_names(input_dir: Path, names: Sequence[str], allow_unsafe_pickle: bool = False) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    path = find_first_existing(input_dir, names)
    if path is None:
        return None, None
    obj = torch_load(path, allow_unsafe_pickle=allow_unsafe_pickle)
    if isinstance(obj, dict):
        # This branch is useful for activation_counts_combined.pt, not usually for score tensors.
        for key in ["activation_probs", "probabilities", "over_zero", "n"]:
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key], path.name + f":{key}"
        raise ValueError(f"{path} is a dict, but no known tensor key was found.")
    if not torch.is_tensor(obj):
        raise ValueError(f"{path} did not contain a tensor; got {type(obj)}")
    return obj, path.name


def normalize_lhd(tensor: Optional[torch.Tensor], num_dialects: int, name: str) -> Optional[torch.Tensor]:
    """Return tensor in [L, H, D] format when possible."""
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise ValueError(f"{name} must be a 3D tensor; got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] == num_dialects:
        return tensor.detach().cpu()
    if tensor.shape[0] == num_dialects:
        return tensor.permute(1, 2, 0).contiguous().detach().cpu()
    raise ValueError(
        f"Could not infer dialect axis for {name} with shape {tuple(tensor.shape)} and {num_dialects} dialects. "
        "Expected [L,H,D] or [D,L,H]."
    )


def normalize_lh(tensor: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be a 2D tensor [L,H]; got shape {tuple(tensor.shape)}")
    return tensor.detach().cpu()


def tensor_finite_flat(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().flatten().to(dtype=torch.float32)
    return y[torch.isfinite(y)]


def distribution_stats(x: Optional[torch.Tensor], name: str) -> Dict[str, Any]:
    if x is None:
        return {"name": name, "available": False}
    y = tensor_finite_flat(x)
    if y.numel() == 0:
        return {"name": name, "available": True, "count": 0}
    qs = torch.tensor([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    vals = torch.quantile(y, qs).tolist()
    std = float(y.std(unbiased=False).item()) if y.numel() > 1 else 0.0
    return {
        "name": name,
        "available": True,
        "count": int(y.numel()),
        "min": float(vals[0]),
        "p01": float(vals[1]),
        "p05": float(vals[2]),
        "p10": float(vals[3]),
        "p25": float(vals[4]),
        "median": float(vals[5]),
        "p75": float(vals[6]),
        "p90": float(vals[7]),
        "p95": float(vals[8]),
        "p99": float(vals[9]),
        "max": float(vals[10]),
        "mean": float(y.mean().item()),
        "std": std,
    }


def read_selected_csv(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rec: Dict[str, Any] = dict(row)
            for key in ["dialect_id", "layer", "neuron"]:
                rec[key] = parse_int(rec.get(key))
            for key in ["entropy", "lape", "normalized_lape", "activation_probability"]:
                if key in rec:
                    rec[key] = parse_float(rec.get(key))
            for key in list(rec.keys()):
                if key.startswith("p_"):
                    rec[key] = parse_float(rec.get(key))
            if rec.get("layer") is not None and rec.get("neuron") is not None:
                records.append(rec)
    return records


def read_selected_json(path: Path) -> List[Dict[str, Any]]:
    obj = read_json(path, None)
    if obj is None:
        return []
    records: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for rec in obj:
            if isinstance(rec, dict):
                records.append(dict(rec))
    elif isinstance(obj, dict):
        # Earlier script format may be {dialect: [records...]}
        for dialect, entries in obj.items():
            if isinstance(entries, list):
                for rec in entries:
                    if isinstance(rec, dict):
                        row = dict(rec)
                        row.setdefault("dialect", dialect)
                        records.append(row)
    for rec in records:
        rec["layer"] = parse_int(rec.get("layer"))
        rec["neuron"] = parse_int(rec.get("neuron"))
        rec["dialect_id"] = parse_int(rec.get("dialect_id"))
        for key in ["entropy", "lape", "normalized_lape", "activation_probability"]:
            if key in rec:
                rec[key] = parse_float(rec.get(key))
    return [r for r in records if r.get("layer") is not None and r.get("neuron") is not None]


def selected_from_neurons_pth(
    input_dir: Path,
    dialects: Sequence[str],
    entropy: Optional[torch.Tensor],
    activation_probs: Optional[torch.Tensor],
    allow_unsafe_pickle: bool,
) -> List[Dict[str, Any]]:
    path = input_dir / "neurons.pth"
    if not path.exists():
        return []
    obj = torch_load(path, allow_unsafe_pickle=allow_unsafe_pickle)
    records: List[Dict[str, Any]] = []
    if not isinstance(obj, list):
        raise ValueError(f"neurons.pth should be a list; got {type(obj)}")
    for d_id, layer_lists in enumerate(obj):
        dialect = dialects[d_id] if d_id < len(dialects) else str(d_id)
        if not isinstance(layer_lists, list):
            continue
        for layer_id, ids in enumerate(layer_lists):
            if torch.is_tensor(ids):
                neuron_ids = [int(x) for x in ids.detach().cpu().flatten().tolist()]
            elif isinstance(ids, (list, tuple)):
                neuron_ids = [int(x) for x in ids]
            else:
                continue
            for neuron_id in neuron_ids:
                rec: Dict[str, Any] = {
                    "dialect": dialect,
                    "dialect_id": d_id,
                    "layer": int(layer_id),
                    "neuron": int(neuron_id),
                }
                if entropy is not None and layer_id < entropy.shape[0] and neuron_id < entropy.shape[1]:
                    rec["entropy"] = float(entropy[layer_id, neuron_id].item())
                if activation_probs is not None and layer_id < activation_probs.shape[0] and neuron_id < activation_probs.shape[1]:
                    rec["activation_probability"] = float(activation_probs[layer_id, neuron_id, d_id].item())
                    for k, d in enumerate(dialects):
                        rec[f"p_{d}"] = float(activation_probs[layer_id, neuron_id, k].item())
                records.append(rec)
    return records


def load_selected_records(
    input_dir: Path,
    dialects: Sequence[str],
    entropy: Optional[torch.Tensor],
    activation_probs: Optional[torch.Tensor],
    allow_unsafe_pickle: bool,
) -> List[Dict[str, Any]]:
    records = read_selected_csv(input_dir / "selected_neurons.csv")
    if not records:
        records = read_selected_json(input_dir / "selected_neurons.json")
    if not records:
        records = selected_from_neurons_pth(input_dir, dialects, entropy, activation_probs, allow_unsafe_pickle)

    dialect_to_id = {d: i for i, d in enumerate(dialects)}
    for rec in records:
        d = rec.get("dialect")
        if d is not None and rec.get("dialect_id") is None and d in dialect_to_id:
            rec["dialect_id"] = dialect_to_id[d]
        d_id = rec.get("dialect_id")
        if d is None and isinstance(d_id, int) and 0 <= d_id < len(dialects):
            rec["dialect"] = dialects[d_id]
        # Fill in score columns from tensors when possible.
        layer = rec.get("layer")
        neuron = rec.get("neuron")
        if isinstance(layer, int) and isinstance(neuron, int):
            if entropy is not None and "entropy" not in rec and layer < entropy.shape[0] and neuron < entropy.shape[1]:
                rec["entropy"] = float(entropy[layer, neuron].item())
            d_id2 = rec.get("dialect_id")
            if activation_probs is not None and isinstance(d_id2, int) and layer < activation_probs.shape[0] and neuron < activation_probs.shape[1]:
                if "activation_probability" not in rec:
                    rec["activation_probability"] = float(activation_probs[layer, neuron, d_id2].item())
                for k, dialect in enumerate(dialects):
                    key = f"p_{dialect}"
                    if key not in rec:
                        rec[key] = float(activation_probs[layer, neuron, k].item())
    return records


def maybe_normalized_entropy(entropy: Optional[torch.Tensor], num_dialects: int) -> Optional[torch.Tensor]:
    if entropy is None:
        return None
    denom = math.log(num_dialects) if num_dialects > 1 else 1.0
    if denom <= 0:
        return None
    return entropy / denom


def infer_shapes(
    activation_probs: Optional[torch.Tensor],
    entropy: Optional[torch.Tensor],
    selected_records: Sequence[Dict[str, Any]],
) -> Tuple[int, Optional[int]]:
    if activation_probs is not None:
        return int(activation_probs.shape[0]), int(activation_probs.shape[1])
    if entropy is not None:
        return int(entropy.shape[0]), int(entropy.shape[1])
    max_layer = max((int(r["layer"]) for r in selected_records if r.get("layer") is not None), default=-1)
    return max_layer + 1, None


def build_selected_count_tables(
    selected_records: Sequence[Dict[str, Any]],
    dialects: Sequence[str],
    num_layers: int,
    intermediate_size: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_dialect = Counter()
    by_layer = Counter()
    by_dl = Counter()
    unique_by_dialect: Dict[str, set] = {d: set() for d in dialects}
    unique_global = set()

    for rec in selected_records:
        dialect = str(rec.get("dialect"))
        layer = rec.get("layer")
        neuron = rec.get("neuron")
        if dialect is None or layer is None or neuron is None:
            continue
        layer = int(layer)
        neuron = int(neuron)
        by_dialect[dialect] += 1
        by_layer[layer] += 1
        by_dl[(dialect, layer)] += 1
        unique_global.add((layer, neuron))
        unique_by_dialect.setdefault(dialect, set()).add((layer, neuron))

    counts_by_dialect: List[Dict[str, Any]] = []
    for dialect in dialects:
        total = int(by_dialect[dialect])
        unique_count = len(unique_by_dialect.get(dialect, set()))
        counts_by_dialect.append(
            {
                "dialect": dialect,
                "selected_count": total,
                "unique_layer_neuron_count": unique_count,
            }
        )

    counts_by_layer: List[Dict[str, Any]] = []
    denom = intermediate_size if intermediate_size and intermediate_size > 0 else None
    for layer in range(num_layers):
        total = int(by_layer[layer])
        row: Dict[str, Any] = {
            "layer": layer,
            "selected_count_total": total,
            "selected_fraction_of_layer": (total / denom) if denom else None,
        }
        for dialect in dialects:
            row[f"selected_{dialect}"] = int(by_dl[(dialect, layer)])
        counts_by_layer.append(row)

    counts_by_dialect_layer: List[Dict[str, Any]] = []
    for dialect in dialects:
        for layer in range(num_layers):
            count = int(by_dl[(dialect, layer)])
            counts_by_dialect_layer.append(
                {
                    "dialect": dialect,
                    "layer": layer,
                    "selected_count": count,
                    "selected_fraction_of_layer": (count / denom) if denom else None,
                }
            )

    return counts_by_dialect, counts_by_layer, counts_by_dialect_layer


def create_top_layers_table(
    counts_by_dialect_layer: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    by_dialect: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in counts_by_dialect_layer:
        by_dialect[str(row["dialect"])].append(dict(row))
    out: List[Dict[str, Any]] = []
    for dialect, rows in by_dialect.items():
        ranked = sorted(rows, key=lambda r: (-int(r["selected_count"]), int(r["layer"])))[:top_k]
        for rank, row in enumerate(ranked, start=1):
            out.append(
                {
                    "dialect": dialect,
                    "rank": rank,
                    "layer": int(row["layer"]),
                    "selected_count": int(row["selected_count"]),
                    "selected_fraction_of_layer": row.get("selected_fraction_of_layer"),
                }
            )
    return out


def save_ranked_selected_neurons(
    path_all: Path,
    path_top: Path,
    records: Sequence[Dict[str, Any]],
    dialects: Sequence[str],
    top_k: int,
) -> None:
    def score_tuple(r: Dict[str, Any]) -> Tuple[Any, ...]:
        # Lower entropy/LAPE is better; higher activation is better.
        entropy = r.get("entropy")
        if entropy is None:
            entropy = r.get("lape")
        entropy_sort = float(entropy) if entropy is not None else float("inf")
        activation = r.get("activation_probability")
        activation_sort = -float(activation) if activation is not None else float("inf")
        return (str(r.get("dialect", "")), entropy_sort, activation_sort, int(r.get("layer") or 0), int(r.get("neuron") or 0))

    ranked = sorted((dict(r) for r in records), key=score_tuple)
    fieldnames = ["dialect", "dialect_id", "layer", "neuron", "entropy", "lape", "normalized_lape", "activation_probability"]
    for d in dialects:
        fieldnames.append(f"p_{d}")
    # Include any extra keys at the end.
    extra_keys = []
    seen = set(fieldnames)
    for r in ranked:
        for k in r.keys():
            if k not in seen and not isinstance(r.get(k), (dict, list)):
                seen.add(k)
                extra_keys.append(k)
    fieldnames = fieldnames + extra_keys
    write_csv(path_all, ranked, fieldnames=fieldnames)
    write_csv(path_top, ranked[:top_k], fieldnames=fieldnames)


def entropy_by_layer_rows(entropy: Optional[torch.Tensor], normalized_entropy: Optional[torch.Tensor]) -> List[Dict[str, Any]]:
    if entropy is None:
        return []
    rows: List[Dict[str, Any]] = []
    for layer in range(entropy.shape[0]):
        row = distribution_stats(entropy[layer], f"entropy_layer_{layer}")
        row["layer"] = layer
        if normalized_entropy is not None:
            nrow = distribution_stats(normalized_entropy[layer], f"normalized_entropy_layer_{layer}")
            for key in ["min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max", "mean", "std"]:
                row[f"normalized_{key}"] = nrow.get(key)
        rows.append(row)
    return rows


def activation_by_dialect_rows(activation_probs: Optional[torch.Tensor], dialects: Sequence[str]) -> List[Dict[str, Any]]:
    if activation_probs is None:
        return []
    rows: List[Dict[str, Any]] = []
    for d_id, dialect in enumerate(dialects):
        row = distribution_stats(activation_probs[:, :, d_id], f"activation_probability_{dialect}")
        row["dialect"] = dialect
        row["dialect_id"] = d_id
        rows.append(row)
    return rows


def activation_by_layer_dialect_rows(activation_probs: Optional[torch.Tensor], dialects: Sequence[str]) -> List[Dict[str, Any]]:
    if activation_probs is None:
        return []
    rows: List[Dict[str, Any]] = []
    for d_id, dialect in enumerate(dialects):
        for layer in range(activation_probs.shape[0]):
            row = distribution_stats(activation_probs[layer, :, d_id], f"activation_probability_{dialect}_layer_{layer}")
            row["dialect"] = dialect
            row["dialect_id"] = d_id
            row["layer"] = layer
            rows.append(row)
    return rows


def selected_score_rows(records: Sequence[Dict[str, Any]], dialects: Sequence[str]) -> List[Dict[str, Any]]:
    by_dialect: Dict[str, List[Dict[str, Any]]] = {d: [] for d in dialects}
    for rec in records:
        d = str(rec.get("dialect"))
        by_dialect.setdefault(d, []).append(dict(rec))

    rows: List[Dict[str, Any]] = []
    for dialect in dialects:
        recs = by_dialect.get(dialect, [])
        for key in ["entropy", "lape", "normalized_lape", "activation_probability"]:
            vals = [parse_float(r.get(key)) for r in recs]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if vals:
                t = torch.tensor(vals, dtype=torch.float32)
                row = distribution_stats(t, f"selected_{key}_{dialect}")
                row["dialect"] = dialect
                row["metric"] = key
                rows.append(row)
    return rows


def plot_selected_counts_by_dialect(path: Path, counts_by_dialect: Sequence[Dict[str, Any]]) -> None:
    dialects = [str(r["dialect"]) for r in counts_by_dialect]
    counts = [int(r["selected_count"]) for r in counts_by_dialect]
    plt.figure(figsize=(max(7, len(dialects) * 1.2), 4.5))
    plt.bar(dialects, counts)
    plt.xlabel("Dialect")
    plt.ylabel("Selected neurons")
    plt.title("Selected neurons by dialect")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_selected_counts_by_layer(path: Path, counts_by_layer: Sequence[Dict[str, Any]], dialects: Sequence[str]) -> None:
    layers = [int(r["layer"]) for r in counts_by_layer]
    plt.figure(figsize=(max(9, len(layers) * 0.28), 5))
    bottom = [0 for _ in layers]
    for dialect in dialects:
        vals = [int(r.get(f"selected_{dialect}", 0)) for r in counts_by_layer]
        plt.bar(layers, vals, bottom=bottom, label=dialect)
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.xlabel("Layer")
    plt.ylabel("Selected neurons")
    plt.title("Selected-neuron distribution by layer")
    if len(dialects) <= 12:
        plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_heatmap_selected_by_dialect_layer(path: Path, counts_by_dialect_layer: Sequence[Dict[str, Any]], dialects: Sequence[str], num_layers: int) -> None:
    matrix = torch.zeros((len(dialects), num_layers), dtype=torch.float32)
    dialect_to_id = {d: i for i, d in enumerate(dialects)}
    for row in counts_by_dialect_layer:
        d = str(row["dialect"])
        layer = int(row["layer"])
        if d in dialect_to_id and 0 <= layer < num_layers:
            matrix[dialect_to_id[d], layer] = float(row["selected_count"])
    plt.figure(figsize=(max(9, num_layers * 0.3), max(3.5, len(dialects) * 0.55)))
    plt.imshow(matrix.numpy(), aspect="auto")
    plt.colorbar(label="Selected neurons")
    plt.yticks(range(len(dialects)), dialects)
    plt.xticks(range(num_layers), [str(i) for i in range(num_layers)], rotation=90)
    plt.xlabel("Layer")
    plt.ylabel("Dialect")
    plt.title("Selected neurons per dialect and layer")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_entropy_hist(path: Path, entropy: Optional[torch.Tensor], normalized: bool = False, bins: int = 80) -> None:
    if entropy is None:
        return
    y = tensor_finite_flat(entropy)
    if y.numel() == 0:
        return
    plt.figure(figsize=(7.5, 4.8))
    plt.hist(y.numpy(), bins=bins)
    plt.xlabel("Normalized entropy" if normalized else "Entropy")
    plt.ylabel("Neuron count")
    plt.title("Normalized LAPE entropy distribution" if normalized else "LAPE entropy distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_entropy_by_layer_boxplot(path: Path, entropy: Optional[torch.Tensor], sample_per_layer: int) -> None:
    if entropy is None:
        return
    data: List[List[float]] = []
    labels: List[str] = []
    generator = torch.Generator().manual_seed(13)
    for layer in range(entropy.shape[0]):
        vals = entropy[layer].detach().flatten().to(dtype=torch.float32)
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            data.append([])
        elif sample_per_layer and sample_per_layer > 0 and vals.numel() > sample_per_layer:
            idx = torch.randperm(vals.numel(), generator=generator)[:sample_per_layer]
            data.append(vals[idx].tolist())
        else:
            data.append(vals.tolist())
        labels.append(str(layer))
    if not any(len(x) for x in data):
        return
    plt.figure(figsize=(max(10, entropy.shape[0] * 0.32), 5.2))
    plt.boxplot(data, showfliers=False)
    plt.xlabel("Layer")
    plt.ylabel("Entropy")
    plt.title("Entropy distribution by layer")
    plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_activation_hist_global(path: Path, activation_probs: Optional[torch.Tensor], bins: int = 80) -> None:
    if activation_probs is None:
        return
    y = tensor_finite_flat(activation_probs)
    if y.numel() == 0:
        return
    plt.figure(figsize=(7.5, 4.8))
    plt.hist(y.numpy(), bins=bins)
    plt.xlabel("Activation probability")
    plt.ylabel("Count")
    plt.title("Global activation-probability distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_activation_hist_by_dialect(path: Path, activation_probs: Optional[torch.Tensor], dialects: Sequence[str], bins: int = 80) -> None:
    if activation_probs is None:
        return
    plt.figure(figsize=(7.8, 5.0))
    any_plotted = False
    for d_id, dialect in enumerate(dialects):
        y = tensor_finite_flat(activation_probs[:, :, d_id])
        if y.numel() == 0:
            continue
        plt.hist(y.numpy(), bins=bins, histtype="step", density=True, label=dialect)
        any_plotted = True
    if not any_plotted:
        plt.close()
        return
    plt.xlabel("Activation probability")
    plt.ylabel("Density")
    plt.title("Activation-probability distributions by dialect")
    if len(dialects) <= 12:
        plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_mean_activation_by_layer(path: Path, activation_probs: Optional[torch.Tensor], dialects: Sequence[str]) -> None:
    if activation_probs is None:
        return
    num_layers = activation_probs.shape[0]
    layers = list(range(num_layers))
    plt.figure(figsize=(max(9, num_layers * 0.28), 5.0))
    for d_id, dialect in enumerate(dialects):
        vals = []
        for layer in range(num_layers):
            finite = tensor_finite_flat(activation_probs[layer, :, d_id])
            vals.append(float(finite.mean().item()) if finite.numel() else float("nan"))
        plt.plot(layers, vals, marker="o", label=dialect)
    plt.xlabel("Layer")
    plt.ylabel("Mean activation probability")
    plt.title("Mean activation probability by layer")
    if len(dialects) <= 12:
        plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_selected_entropy_vs_activation(path: Path, records: Sequence[Dict[str, Any]], dialects: Sequence[str]) -> None:
    if not records:
        return
    dialect_to_x: Dict[str, Tuple[List[float], List[float]]] = {d: ([], []) for d in dialects}
    for rec in records:
        d = str(rec.get("dialect"))
        entropy = rec.get("entropy")
        if entropy is None:
            entropy = rec.get("lape")
        activation = rec.get("activation_probability")
        entropy_f = parse_float(entropy)
        activation_f = parse_float(activation)
        if entropy_f is None or activation_f is None:
            continue
        if not (math.isfinite(entropy_f) and math.isfinite(activation_f)):
            continue
        if d not in dialect_to_x:
            dialect_to_x[d] = ([], [])
        dialect_to_x[d][0].append(entropy_f)
        dialect_to_x[d][1].append(activation_f)
    if not any(len(xy[0]) for xy in dialect_to_x.values()):
        return
    plt.figure(figsize=(7.8, 5.2))
    for dialect, (xs, ys) in dialect_to_x.items():
        if xs:
            plt.scatter(xs, ys, label=dialect, alpha=0.75)
    plt.xlabel("Entropy / LAPE")
    plt.ylabel("Activation probability for selected dialect")
    plt.title("Selected neurons: entropy vs activation probability")
    if len(dialect_to_x) <= 12:
        plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_unique_overlap_matrix(path: Path, records: Sequence[Dict[str, Any]], dialects: Sequence[str]) -> List[Dict[str, Any]]:
    sets: Dict[str, set] = {d: set() for d in dialects}
    for rec in records:
        d = str(rec.get("dialect"))
        layer = rec.get("layer")
        neuron = rec.get("neuron")
        if d in sets and layer is not None and neuron is not None:
            sets[d].add((int(layer), int(neuron)))
    matrix = torch.zeros((len(dialects), len(dialects)), dtype=torch.float32)
    rows: List[Dict[str, Any]] = []
    for i, d1 in enumerate(dialects):
        for j, d2 in enumerate(dialects):
            inter = len(sets[d1].intersection(sets[d2]))
            union = len(sets[d1].union(sets[d2]))
            jaccard = inter / union if union else 0.0
            matrix[i, j] = jaccard
            rows.append({"dialect_a": d1, "dialect_b": d2, "intersection": inter, "union": union, "jaccard": jaccard})
    plt.figure(figsize=(max(4.5, len(dialects) * 0.7), max(4.0, len(dialects) * 0.6)))
    plt.imshow(matrix.numpy(), aspect="equal", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Jaccard overlap")
    plt.xticks(range(len(dialects)), dialects, rotation=45, ha="right")
    plt.yticks(range(len(dialects)), dialects)
    plt.title("Selected-neuron overlap between dialects")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return rows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_dir / "analysis"
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    dialects = infer_dialects(input_dir)
    log(f"Dialects: {', '.join(dialects)}")

    activation_raw, activation_source = load_tensor_by_names(
        input_dir,
        ["activation_probs.pt", "activation_probabilities.pt"],
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    activation_probs = normalize_lhd(activation_raw, len(dialects), activation_source or "activation_probs") if activation_raw is not None else None

    entropy_raw, entropy_source = load_tensor_by_names(
        input_dir,
        ["entropy.pt", "lape_scores.pt"],
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    entropy = normalize_lh(entropy_raw, entropy_source or "entropy") if entropy_raw is not None else None

    entropy_selection_raw, entropy_selection_source = load_tensor_by_names(
        input_dir,
        ["entropy_for_selection.pt"],
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    entropy_for_selection = normalize_lh(entropy_selection_raw, entropy_selection_source or "entropy_for_selection") if entropy_selection_raw is not None else None

    normalized_entropy_raw, normalized_entropy_source = load_tensor_by_names(
        input_dir,
        ["normalized_lape_scores.pt"],
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    normalized_entropy = normalize_lh(normalized_entropy_raw, normalized_entropy_source or "normalized_entropy") if normalized_entropy_raw is not None else maybe_normalized_entropy(entropy, len(dialects))

    active_counts_raw, active_counts_source = load_tensor_by_names(
        input_dir,
        ["over_zero.pt", "active_counts.pt"],
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    active_counts = normalize_lhd(active_counts_raw, len(dialects), active_counts_source or "active_counts") if active_counts_raw is not None else None

    token_counts_obj = torch_load(input_dir / "token_counts.pt", args.allow_unsafe_pickle) if (input_dir / "token_counts.pt").exists() else None
    token_counts: Optional[List[int]] = None
    if torch.is_tensor(token_counts_obj):
        token_counts = [int(x) for x in token_counts_obj.detach().cpu().flatten().tolist()]
    else:
        token_counts_json = read_json(input_dir / "token_counts.json", None)
        if isinstance(token_counts_json, dict):
            token_counts = [int(token_counts_json.get(d, 0)) for d in dialects]

    selected_records = load_selected_records(input_dir, dialects, entropy, activation_probs, args.allow_unsafe_pickle)
    log(f"Loaded {len(selected_records)} selected-neuron records")

    num_layers, intermediate_size = infer_shapes(activation_probs, entropy, selected_records)
    counts_by_dialect, counts_by_layer, counts_by_dialect_layer = build_selected_count_tables(
        selected_records, dialects, num_layers, intermediate_size
    )
    top_layers = create_top_layers_table(counts_by_dialect_layer, args.top_k_layers)

    # Tables.
    write_csv(tables_dir / "selected_counts_by_dialect.csv", counts_by_dialect)
    write_csv(tables_dir / "selected_counts_by_layer.csv", counts_by_layer)
    write_csv(tables_dir / "selected_counts_by_dialect_layer.csv", counts_by_dialect_layer)
    write_csv(tables_dir / "top_layers_by_dialect.csv", top_layers)
    save_ranked_selected_neurons(
        tables_dir / "selected_neurons_ranked.csv",
        tables_dir / "selected_neurons_ranked_topk.csv",
        selected_records,
        dialects,
        args.top_k_neurons,
    )

    entropy_layer_table = entropy_by_layer_rows(entropy, normalized_entropy)
    if entropy_layer_table:
        write_csv(tables_dir / "entropy_distribution_by_layer.csv", entropy_layer_table)
    act_dialect_table = activation_by_dialect_rows(activation_probs, dialects)
    if act_dialect_table:
        write_csv(tables_dir / "activation_probability_distribution_by_dialect.csv", act_dialect_table)
    act_layer_dialect_table = activation_by_layer_dialect_rows(activation_probs, dialects)
    if act_layer_dialect_table:
        write_csv(tables_dir / "activation_probability_distribution_by_layer_dialect.csv", act_layer_dialect_table)
    selected_score_table = selected_score_rows(selected_records, dialects)
    if selected_score_table:
        write_csv(tables_dir / "selected_neuron_score_distributions.csv", selected_score_table)

    # Plots.
    plot_selected_counts_by_dialect(plots_dir / "selected_counts_by_dialect.png", counts_by_dialect)
    plot_selected_counts_by_layer(plots_dir / "selected_counts_by_layer.png", counts_by_layer, dialects)
    plot_heatmap_selected_by_dialect_layer(
        plots_dir / "selected_counts_heatmap_dialect_layer.png",
        counts_by_dialect_layer,
        dialects,
        num_layers,
    )
    plot_entropy_hist(plots_dir / "entropy_histogram.png", entropy, normalized=False, bins=args.hist_bins)
    plot_entropy_hist(plots_dir / "normalized_entropy_histogram.png", normalized_entropy, normalized=True, bins=args.hist_bins)
    plot_entropy_by_layer_boxplot(plots_dir / "entropy_by_layer_boxplot.png", entropy, args.sample_per_layer_boxplot)
    plot_activation_hist_global(plots_dir / "activation_probability_histogram_global.png", activation_probs, bins=args.hist_bins)
    plot_activation_hist_by_dialect(plots_dir / "activation_probability_histogram_by_dialect.png", activation_probs, dialects, bins=args.hist_bins)
    plot_mean_activation_by_layer(plots_dir / "mean_activation_probability_by_layer.png", activation_probs, dialects)
    plot_selected_entropy_vs_activation(plots_dir / "selected_entropy_vs_activation.png", selected_records, dialects)
    overlap_rows = plot_unique_overlap_matrix(plots_dir / "selected_neuron_overlap_jaccard.png", selected_records, dialects)
    write_csv(tables_dir / "selected_neuron_overlap_jaccard.csv", overlap_rows)

    # Summary JSON.
    selected_total = sum(int(r["selected_count"]) for r in counts_by_dialect)
    unique_selected = len({(int(r["layer"]), int(r["neuron"])) for r in selected_records if r.get("layer") is not None and r.get("neuron") is not None})
    per_layer_counts = [int(r["selected_count_total"]) for r in counts_by_layer]
    max_layer_count = max(per_layer_counts) if per_layer_counts else 0
    max_layers = [int(r["layer"]) for r in counts_by_layer if int(r["selected_count_total"]) == max_layer_count]

    summary = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "dialects": dialects,
        "num_dialects": len(dialects),
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "token_counts": {dialects[i]: token_counts[i] for i in range(min(len(dialects), len(token_counts or [])))} if token_counts else None,
        "selected_total_records": selected_total,
        "selected_unique_layer_neuron_pairs": unique_selected,
        "selected_counts_by_dialect": {r["dialect"]: int(r["selected_count"]) for r in counts_by_dialect},
        "layer_with_max_selected_count": max_layers,
        "max_selected_count_in_one_layer": max_layer_count,
        "entropy_distribution": distribution_stats(entropy, "entropy"),
        "normalized_entropy_distribution": distribution_stats(normalized_entropy, "normalized_entropy"),
        "entropy_for_selection_distribution": distribution_stats(entropy_for_selection, "entropy_for_selection"),
        "activation_probability_distribution_global": distribution_stats(activation_probs, "activation_probs"),
        "active_counts_distribution_global": distribution_stats(active_counts, "active_counts"),
        "source_files_detected": {
            "activation_probs": activation_source,
            "entropy": entropy_source,
            "entropy_for_selection": entropy_selection_source,
            "normalized_entropy": normalized_entropy_source or ("computed_from_entropy" if normalized_entropy is not None else None),
            "active_counts": active_counts_source,
        },
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
    }
    write_json(out_dir / "analysis_summary.json", summary)

    # Lightweight Markdown report for quick inspection.
    report_lines = [
        "# LAPE output analysis",
        "",
        f"Input directory: `{input_dir}`",
        f"Dialects: {', '.join(dialects)}",
        f"Layers: {num_layers}",
        f"Intermediate size: {intermediate_size if intermediate_size is not None else 'unknown'}",
        f"Selected records: {selected_total}",
        f"Unique selected layer/neuron pairs: {unique_selected}",
        "",
        "## Selected counts by dialect",
        "",
        "| Dialect | Selected count | Unique layer/neuron count |",
        "|---|---:|---:|",
    ]
    for row in counts_by_dialect:
        report_lines.append(f"| {row['dialect']} | {row['selected_count']} | {row['unique_layer_neuron_count']} |")
    report_lines.extend(
        [
            "",
            "## Key plots",
            "",
            "- `plots/selected_counts_by_dialect.png`",
            "- `plots/selected_counts_by_layer.png`",
            "- `plots/selected_counts_heatmap_dialect_layer.png`",
            "- `plots/entropy_histogram.png`",
            "- `plots/activation_probability_histogram_by_dialect.png`",
            "- `plots/mean_activation_probability_by_layer.png`",
            "",
            "## Key tables",
            "",
            "- `tables/selected_counts_by_dialect.csv`",
            "- `tables/selected_counts_by_layer.csv`",
            "- `tables/selected_counts_by_dialect_layer.csv`",
            "- `tables/top_layers_by_dialect.csv`",
            "- `tables/selected_neurons_ranked.csv`",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    log(f"Analysis written to: {out_dir}")
    log(f"Plots: {plots_dir}")
    log(f"Tables: {tables_dir}")


if __name__ == "__main__":
    main()