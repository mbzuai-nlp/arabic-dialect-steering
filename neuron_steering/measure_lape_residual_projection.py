#!/usr/bin/env python3
"""
Measure residual-space coverage by LAPE-selected MLP neuron output directions.

This script bridges two different spaces:

  * existing dialect steering vectors: residual-stream directions, usually
    shaped [num_layers + 1, hidden_size] because hidden_states[0] is embeddings;
  * LAPE neurons: MLP intermediate coordinates, shaped [num_layers, intermediate].

For each transformer layer l, the selected MLP neurons write into residual space
through columns of the MLP down projection. This script projects the residual
dialect vector v_l onto the span of those selected columns and computes:

    coverage_l = ||Proj_span(W_selected_l)(v_l)||_2^2 / ||v_l||_2^2

and the global aggregate:

    coverage = sum_l projected_energy_l / sum_l total_energy_l

Interpretation: this is residual-subspace coverage, not literal MLP activation
energy coverage. It asks how much of the residual dialect direction could be
represented by the output directions of the selected neurons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


MODEL_ALIASES = {
    "allam": "humain-ai/ALLaM-7B-Instruct-preview",
    "allam-7b": "humain-ai/ALLaM-7B-Instruct-preview",
    "allam-7b-instruct-preview": "humain-ai/ALLaM-7B-Instruct-preview",
    "humain-ai/allam-7b-instruct-preview": "humain-ai/ALLaM-7B-Instruct-preview",
    "fanar": "QCRI/Fanar-1-9B",
    "fanar-1-9b": "QCRI/Fanar-1-9B",
    "qcri/fanar-1-9b": "QCRI/Fanar-1-9B",
    "fanar-instruct": "QCRI/Fanar-1-9B-Instruct",
    "fanar-1-9b-instruct": "QCRI/Fanar-1-9B-Instruct",
    "jais2": "inceptionai/Jais-2-8B-Chat",
    "jais2-8b-chat": "inceptionai/Jais-2-8B-Chat",
    "jais-2-8b-chat": "inceptionai/Jais-2-8B-Chat",
    "inceptionai/jais-2-8b-chat": "inceptionai/Jais-2-8B-Chat",
}


SummaryRow = Dict[str, Any]
LayerRow = Dict[str, Any]


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def import_runtime() -> Tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "This script requires torch and transformers. Run it in the same "
            "environment used for neuron steering, e.g. conda run -n steering ..."
        ) from exc
    return torch, nn, AutoModelForCausalLM


def parse_bool_flag(value: Optional[str]) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project a residual dialect vector onto LAPE-selected MLP output directions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model alias or Hugging Face model ID.")
    parser.add_argument("--neurons_dir", required=True, help="LAPE output directory.")
    parser.add_argument("--vector_path", required=True, help="Residual dialect vector .pt file.")
    parser.add_argument("--target_dialect", required=True, help="LAPE dialect label, e.g. CAI.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
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
            "How LAPE layer IDs align to residual vector rows. same uses vector[layer]; "
            "hidden_states uses vector[layer + 1]; auto chooses hidden_states for "
            "num_layers+1 vectors and same for num_layers vectors."
        ),
    )
    parser.add_argument("--layer_offset", type=int, default=None, help="Manual vector row offset.")
    parser.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="Optional debug cap on number of LAPE layers to process.",
    )

    parser.add_argument(
        "--projection_method",
        choices=["svd"],
        default="svd",
        help="Projection method. SVD gives a rank-aware orthonormal basis.",
    )
    parser.add_argument(
        "--svd_rcond",
        type=float,
        default=None,
        help="Relative singular-value cutoff. Default uses max(A.shape) * eps.",
    )
    parser.add_argument(
        "--random_baseline",
        type=int,
        default=0,
        help=(
            "Number of random neuron masks to sample. Each mask uses the same "
            "per-layer neuron counts as the identified LAPE neurons."
        ),
    )
    parser.add_argument(
        "--random_exclude_selected",
        action="store_true",
        help="Sample random baseline neurons from the non-selected neurons in each layer.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for baseline masks.")

    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="dtype passed to from_pretrained.",
    )
    parser.add_argument("--device_map", default="auto", help="HF device_map. Use 'none' to disable.")
    parser.add_argument("--device", default="cpu", help="Manual device when --device_map none.")
    parser.add_argument("--trust_remote_code", type=parse_bool_flag, nargs="?", const=True, default=True)
    parser.add_argument(
        "--allow_unsafe_pickle",
        action="store_true",
        help="Allow torch.load fallback without weights_only=True for trusted local artifacts.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and vector shape, but do not load model weights or compute projections.",
    )
    return parser.parse_args()


def resolve_model_id(model_arg: str) -> str:
    return MODEL_ALIASES.get(model_arg.lower(), model_arg)


def dtype_from_arg(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


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


def torch_load(torch: Any, path: Path, allow_unsafe_pickle: bool = False) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        if allow_unsafe_pickle:
            return torch.load(path, map_location="cpu")
        raise RuntimeError(
            f"Could not safely load {path}. If this is a trusted file, retry with "
            "--allow_unsafe_pickle. Original error: " + repr(exc)
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
            "Tried response_avg_diff, vector, vectors, steering_vector."
        )
    if isinstance(obj, (list, tuple)) and obj and all(torch.is_tensor(x) for x in obj):
        return torch.stack([x.detach().cpu() for x in obj], dim=0)
    raise ValueError(f"{source} did not load as a tensor, tensor dict, or list of tensors.")


def format_shape(obj: Any) -> str:
    shape = getattr(obj, "shape", None)
    if shape is None:
        return ""
    return "[" + ", ".join(str(int(x)) for x in shape) + "]"


def load_lape_shape(neurons_dir: Path) -> Tuple[int, int]:
    run_summary = read_json(neurons_dir / "run_summary.json", {}) or {}
    try:
        num_layers = int(run_summary["num_layers"])
        intermediate_size = int(run_summary["intermediate_size"])
    except Exception as exc:
        raise ValueError(
            f"Could not read num_layers/intermediate_size from {neurons_dir / 'run_summary.json'}."
        ) from exc
    return num_layers, intermediate_size


def load_selected_for_dialect(selected_csv: Path, target_dialect: str) -> Dict[int, Set[int]]:
    if not selected_csv.exists():
        raise FileNotFoundError(f"selected_neurons.csv not found: {selected_csv}")
    selected: Dict[int, Set[int]] = defaultdict(set)
    with selected_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dialect", "layer", "neuron"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{selected_csv} is missing required columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            if str(row.get("dialect", "")).strip() != target_dialect:
                continue
            try:
                layer = int(float(str(row.get("layer", "")).strip()))
                neuron = int(float(str(row.get("neuron", "")).strip()))
            except ValueError as exc:
                raise ValueError(f"Invalid layer/neuron at {selected_csv}:{line_no}: {row}") from exc
            selected[layer].add(neuron)
    return selected


def count_selected(selected_by_layer: Dict[int, Set[int]]) -> int:
    return sum(len(v) for v in selected_by_layer.values())


def processed_layer_count(lape_layers: int, max_layers: Optional[int]) -> int:
    if max_layers is None:
        return lape_layers
    if int(max_layers) < 0:
        raise ValueError("--max_layers must be non-negative.")
    return min(lape_layers, int(max_layers))


def count_selected_in_processed_layers(selected_by_layer: Dict[int, Set[int]], max_layers: int) -> int:
    return sum(len(selected_by_layer.get(layer, set())) for layer in range(max_layers))


def infer_layer_offset(
    vector_layers: int,
    lape_layers: int,
    alignment: str,
    manual_offset: Optional[int],
) -> Tuple[int, str]:
    if manual_offset is not None:
        if manual_offset < 0:
            raise ValueError("--layer_offset must be non-negative.")
        if vector_layers < lape_layers + manual_offset:
            raise ValueError(
                f"Vector has {vector_layers} rows, but LAPE needs {lape_layers} "
                f"layers with offset {manual_offset}."
            )
        return manual_offset, "manual"

    if alignment == "same":
        if vector_layers < lape_layers:
            raise ValueError(f"Vector has {vector_layers} rows, fewer than LAPE layers {lape_layers}.")
        return 0, "same"

    if alignment == "hidden_states":
        if vector_layers < lape_layers + 1:
            raise ValueError(
                f"Vector has {vector_layers} rows, but hidden_states alignment needs at least {lape_layers + 1}."
            )
        return 1, "hidden_states"

    if vector_layers == lape_layers + 1:
        return 1, "auto:hidden_states"
    if vector_layers == lape_layers:
        return 0, "auto:same"
    raise ValueError(
        f"Could not infer layer alignment: vector has {vector_layers} rows, LAPE has {lape_layers} layers. "
        "Use --layer_offset to override."
    )


def find_transformer_layers(model: Any) -> Tuple[str, Sequence[Any]]:
    candidates = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "decoder.layers",
        "transformer.blocks",
        "blocks",
    ]
    for path in candidates:
        current = model
        ok = True
        for part in path.split("."):
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                ok = False
                break
        if ok and hasattr(current, "__len__") and hasattr(current, "__getitem__"):
            return path, current
    raise ValueError("Could not locate transformer layers on model.")


def get_mlp(layer: Any) -> Any:
    for name in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError(f"Could not find MLP module in layer type {type(layer)}")


def get_down_projection(mlp: Any) -> Tuple[str, Any]:
    for name in ["down_proj", "dense_4h_to_h", "c_proj", "fc2"]:
        if hasattr(mlp, name):
            return name, getattr(mlp, name)
    raise ValueError(f"Could not find a supported MLP down projection on {type(mlp)}")


def projection_weight_matrix(torch: Any, projection: Any, hidden_dim: int, intermediate_size: int) -> Any:
    weight = getattr(projection, "weight", None)
    if weight is None:
        raise ValueError(f"Projection {projection} does not expose a .weight tensor.")
    weight = weight.detach().to(device="cpu", dtype=torch.float32)
    if weight.ndim != 2:
        raise ValueError(f"Projection weight must be 2-D, got shape {tuple(weight.shape)}.")

    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    if rows == hidden_dim and cols == intermediate_size:
        return weight
    if rows == intermediate_size and cols == hidden_dim:
        return weight.t().contiguous()
    raise ValueError(
        f"Projection weight shape {tuple(weight.shape)} cannot be interpreted as "
        f"[hidden_dim={hidden_dim}, intermediate_size={intermediate_size}] or its transpose."
    )


def projection_energy_svd(torch: Any, matrix: Any, vector: Any, rcond: Optional[float]) -> Tuple[float, int, float]:
    if matrix.numel() == 0 or matrix.shape[1] == 0:
        return 0.0, 0, 0.0

    u, singular_values, _vh = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0:
        return 0.0, 0, 0.0
    largest = float(singular_values.max().item())
    if largest <= 0.0:
        return 0.0, 0, 0.0

    if rcond is None:
        eps = torch.finfo(singular_values.dtype).eps
        tol = float(max(matrix.shape) * eps * largest)
    else:
        tol = float(rcond * largest)

    rank = int((singular_values > tol).sum().item())
    if rank <= 0:
        return 0.0, 0, tol

    basis = u[:, :rank]
    coeff = basis.t().matmul(vector)
    projected_energy = float(coeff.pow(2).sum().item())
    return projected_energy, rank, tol


def random_neuron_sets(
    rng: random.Random,
    selected_by_layer: Dict[int, Set[int]],
    intermediate_size: int,
    max_layers: int,
    exclude_selected: bool = False,
) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for layer in range(max_layers):
        selected = selected_by_layer.get(layer, set())
        count = len(selected)
        if count <= 0:
            continue
        if exclude_selected:
            pool = [neuron for neuron in range(intermediate_size) if neuron not in selected]
        else:
            pool = range(intermediate_size)
        if count > len(pool):
            raise ValueError(
                f"Cannot sample {count} random neurons for layer {layer} from a pool of {len(pool)}."
            )
        out[layer] = set(rng.sample(pool, count))
    return out


def measure_projection(
    torch: Any,
    model: Any,
    vector: Any,
    selected_by_layer: Dict[int, Set[int]],
    lape_layers: int,
    intermediate_size: int,
    layer_offset: int,
    layer_alignment_note: str,
    args: argparse.Namespace,
    log_layers: bool = True,
) -> Tuple[SummaryRow, List[LayerRow]]:
    layers_name, layers = find_transformer_layers(model)
    if len(layers) < lape_layers:
        raise ValueError(f"Model has {len(layers)} layers at {layers_name}, but LAPE expects {lape_layers}.")

    hidden_dim = int(vector.shape[1])
    total_energy = 0.0
    projected_energy = 0.0
    layer_rows: List[LayerRow] = []
    max_layers = processed_layer_count(lape_layers, args.max_layers)
    selected_neurons = count_selected_in_processed_layers(selected_by_layer, max_layers)

    for layer_id in range(max_layers):
        vector_layer_id = layer_id + layer_offset
        v = vector[vector_layer_id].to(dtype=torch.float32).flatten()
        layer_energy = float(v.pow(2).sum().item())
        selected = sorted(selected_by_layer.get(layer_id, set()))

        if selected:
            bad = [n for n in selected if n < 0 or n >= intermediate_size]
            if bad:
                examples = ", ".join(str(x) for x in bad[:8])
                raise ValueError(f"Layer {layer_id} has selected neuron indices outside bounds: {examples}")

            mlp = get_mlp(layers[layer_id])
            down_name, down_proj = get_down_projection(mlp)
            down_weight = projection_weight_matrix(torch, down_proj, hidden_dim, intermediate_size)
            selected_indices = torch.tensor(selected, dtype=torch.long)
            selected_directions = down_weight.index_select(1, selected_indices)
            layer_projected, rank, svd_tol = projection_energy_svd(
                torch, selected_directions, v, args.svd_rcond
            )
            layer_projected = min(layer_projected, layer_energy)
        else:
            down_name = ""
            layer_projected = 0.0
            rank = 0
            svd_tol = 0.0

        total_energy += layer_energy
        projected_energy += layer_projected
        layer_coverage = layer_projected / layer_energy if layer_energy > 0 else 0.0
        layer_rows.append(
            {
                "layer": layer_id,
                "vector_layer": vector_layer_id,
                "down_projection": down_name,
                "selected_neurons": len(selected),
                "subspace_rank": rank,
                "svd_tol": svd_tol,
                "layer_energy": layer_energy,
                "projected_energy": layer_projected,
                "unexplained_energy": layer_energy - layer_projected,
                "coverage": layer_coverage,
                "coverage_percent": 100.0 * layer_coverage,
                "selected_dimension_fraction": len(selected) / intermediate_size if intermediate_size > 0 else 0.0,
            }
        )
        if log_layers:
            eprint(
                f"Layer {layer_id}: selected={len(selected)} rank={rank} "
                f"coverage={100.0 * layer_coverage:.4f}%"
            )

    coverage = projected_energy / total_energy if total_energy > 0 else 0.0
    summary: SummaryRow = {
        "status": "ok",
        "reason": "",
        "target_dialect": args.target_dialect,
        "model_id": resolve_model_id(args.model),
        "vector_path": str(Path(args.vector_path).resolve()),
        "vector_shape": format_shape(vector),
        "layers_path": layers_name,
        "layer_alignment": layer_alignment_note,
        "layer_offset": layer_offset,
        "lape_layers": lape_layers,
        "processed_layers": max_layers,
        "hidden_dim": hidden_dim,
        "intermediate_size": intermediate_size,
        "selected_neurons": selected_neurons,
        "total_dimensions": max_layers * intermediate_size,
        "selected_dimension_fraction": selected_neurons / (max_layers * intermediate_size)
        if max_layers * intermediate_size > 0
        else 0.0,
        "total_energy": total_energy,
        "projected_energy": projected_energy,
        "unexplained_energy": total_energy - projected_energy,
        "coverage": coverage,
        "coverage_percent": 100.0 * coverage,
        "projection_method": args.projection_method,
        "svd_rcond": args.svd_rcond if args.svd_rcond is not None else "",
    }
    return summary, layer_rows


def measure_random_baseline(
    torch: Any,
    model: Any,
    vector: Any,
    selected_by_layer: Dict[int, Set[int]],
    lape_layers: int,
    intermediate_size: int,
    layer_offset: int,
    total_energy: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if args.random_baseline <= 0:
        return {}, []

    rng = random.Random(args.seed)
    samples: List[float] = []
    sample_rows: List[Dict[str, Any]] = []
    max_layers = processed_layer_count(lape_layers, args.max_layers)
    random_neurons = count_selected_in_processed_layers(selected_by_layer, max_layers)

    for i in range(int(args.random_baseline)):
        eprint(f"Random baseline {i + 1}/{args.random_baseline}")
        random_selected = random_neuron_sets(
            rng,
            selected_by_layer,
            intermediate_size,
            max_layers=max_layers,
            exclude_selected=args.random_exclude_selected,
        )
        random_summary, _rows = measure_projection(
            torch=torch,
            model=model,
            vector=vector,
            selected_by_layer=random_selected,
            lape_layers=lape_layers,
            intermediate_size=intermediate_size,
            layer_offset=layer_offset,
            layer_alignment_note="random_baseline",
            args=args,
            log_layers=False,
        )
        projected = float(random_summary["projected_energy"])
        coverage = projected / total_energy if total_energy > 0 else 0.0
        samples.append(projected)
        sample_rows.append(
            {
                "sample": i + 1,
                "seed": args.seed,
                "random_neurons": random_neurons,
                "projected_energy": projected,
                "coverage": coverage,
                "coverage_percent": 100.0 * coverage,
            }
        )

    if not samples:
        return {}, []

    ordered = sorted(samples)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = p * (len(ordered) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    mean = statistics.fmean(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    coverage_mean = mean / total_energy if total_energy > 0 else 0.0
    coverage_std = std / total_energy if total_energy > 0 else 0.0
    return {
        "random_sampling": "same_per_layer_count",
        "random_exclude_selected": bool(args.random_exclude_selected),
        "random_neurons_per_sample": random_neurons,
        "random_samples": len(samples),
        "random_projected_energy_mean": mean,
        "random_projected_energy_std": std,
        "random_projected_energy_p05": percentile(0.05),
        "random_projected_energy_p50": percentile(0.50),
        "random_projected_energy_p95": percentile(0.95),
        "random_coverage_mean": coverage_mean,
        "random_coverage_std": coverage_std,
        "random_coverage_p05": percentile(0.05) / total_energy if total_energy > 0 else 0.0,
        "random_coverage_p50": percentile(0.50) / total_energy if total_energy > 0 else 0.0,
        "random_coverage_p95": percentile(0.95) / total_energy if total_energy > 0 else 0.0,
    }, sample_rows


def load_model(torch: Any, AutoModelForCausalLM: Any, args: argparse.Namespace) -> Any:
    model_id = resolve_model_id(args.model)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype_from_arg(torch, args.dtype),
    }
    if args.device_map.lower() != "none":
        kwargs["device_map"] = args.device_map

    eprint(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if args.device_map.lower() == "none":
        model.to(args.device)
    model.eval()
    return model


def summary_fieldnames(include_random: bool) -> List[str]:
    fields = [
        "status",
        "reason",
        "target_dialect",
        "model_id",
        "vector_path",
        "vector_shape",
        "layers_path",
        "layer_alignment",
        "layer_offset",
        "lape_layers",
        "processed_layers",
        "hidden_dim",
        "intermediate_size",
        "selected_neurons",
        "total_dimensions",
        "selected_dimension_fraction",
        "total_energy",
        "projected_energy",
        "unexplained_energy",
        "coverage",
        "coverage_percent",
        "projection_method",
        "svd_rcond",
    ]
    if include_random:
        fields.extend(
            [
                "random_sampling",
                "random_exclude_selected",
                "random_neurons_per_sample",
                "random_samples",
                "random_projected_energy_mean",
                "random_projected_energy_std",
                "random_projected_energy_p05",
                "random_projected_energy_p50",
                "random_projected_energy_p95",
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
        "layer",
        "vector_layer",
        "down_projection",
        "selected_neurons",
        "subspace_rank",
        "svd_tol",
        "layer_energy",
        "projected_energy",
        "unexplained_energy",
        "coverage",
        "coverage_percent",
        "selected_dimension_fraction",
    ]


def random_sample_fieldnames() -> List[str]:
    return [
        "sample",
        "seed",
        "random_neurons",
        "projected_energy",
        "coverage",
        "coverage_percent",
    ]


def write_report(path: Path, summary: SummaryRow, layer_rows: Sequence[LayerRow], include_random: bool) -> None:
    lines = [
        "# LAPE residual-subspace projection",
        "",
        f"Target dialect: `{summary.get('target_dialect', '')}`",
        f"Model: `{summary.get('model_id', '')}`",
        f"Vector: `{summary.get('vector_path', '')}`",
        f"Vector shape: `{summary.get('vector_shape', '')}`",
        f"Layer alignment: `{summary.get('layer_alignment', '')}`",
        "",
    ]

    if summary.get("status") == "ok":
        lines.extend(
            [
                "## Summary",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Residual-subspace coverage | {float(summary['coverage_percent']):.4f}% |",
                f"| Total residual-vector energy | {float(summary['total_energy']):.6g} |",
                f"| Projected energy | {float(summary['projected_energy']):.6g} |",
                f"| Selected neurons | {int(summary['selected_neurons'])} |",
                f"| Selected dimension share | {100.0 * float(summary['selected_dimension_fraction']):.4f}% |",
                "",
                "## Layer Coverage",
                "",
                "| Layer | Vector row | Selected neurons | Rank | Coverage |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in layer_rows:
            lines.append(
                f"| {row['layer']} | {row['vector_layer']} | {row['selected_neurons']} | "
                f"{row['subspace_rank']} | {float(row['coverage_percent']):.4f}% |"
            )
        lines.append("")

        if include_random and "random_coverage_mean" in summary:
            z = summary.get("coverage_random_z", "")
            z_text = "" if z == "" else f"{float(z):.3f}"
            lines.extend(
                [
                    "## Random Baseline",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                    f"| Random samples | {summary.get('random_samples', '')} |",
                    f"| Random neurons per sample | {summary.get('random_neurons_per_sample', '')} |",
                    f"| Random sampling | `{summary.get('random_sampling', '')}` |",
                    f"| Excluded selected neurons | {summary.get('random_exclude_selected', '')} |",
                    f"| Random coverage mean | {100.0 * float(summary['random_coverage_mean']):.4f}% |",
                    f"| Random coverage p05-p95 | {100.0 * float(summary['random_coverage_p05']):.4f}% - {100.0 * float(summary['random_coverage_p95']):.4f}% |",
                    f"| LAPE minus random mean | {100.0 * float(summary['coverage_minus_random_mean']):.4f}% |",
                    f"| Z score | {z_text} |",
                    "",
                ]
            )

        lines.extend(
            [
                "## Interpretation Note",
                "",
                "This is not literal MLP-neuron activation energy coverage.",
                "It measures how much of the residual dialect vector lies in the residual subspace spanned by the selected neurons' MLP output directions.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Not Computed",
                "",
                f"Status: `{summary.get('status', '')}`",
                "",
                f"Reason: {summary.get('reason', '')}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch, _nn, AutoModelForCausalLM = import_runtime()

    neurons_dir = Path(args.neurons_dir).expanduser().resolve()
    vector_path = Path(args.vector_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    selected_csv = Path(args.selected_csv).expanduser().resolve() if args.selected_csv else neurons_dir / "selected_neurons.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    lape_layers, intermediate_size = load_lape_shape(neurons_dir)
    selected_by_layer = load_selected_for_dialect(selected_csv, args.target_dialect)
    selected_count = count_selected(selected_by_layer)
    if selected_count == 0:
        raise ValueError(f"No selected neurons found for dialect {args.target_dialect!r} in {selected_csv}.")

    vector = normalize_vector_object(
        torch,
        torch_load(torch, vector_path, args.allow_unsafe_pickle),
        vector_path,
    ).to(dtype=torch.float32)
    if vector.ndim != 2:
        raise ValueError(f"Expected vector tensor shape [layers, hidden_dim], got {format_shape(vector)}.")

    layer_offset, layer_alignment_note = infer_layer_offset(
        vector_layers=int(vector.shape[0]),
        lape_layers=lape_layers,
        alignment=args.layer_alignment,
        manual_offset=args.layer_offset,
    )

    eprint(f"LAPE shape: {lape_layers} layers x {intermediate_size} MLP neurons")
    eprint(f"Target dialect: {args.target_dialect}; selected neurons: {selected_count}")
    eprint(f"Residual vector shape: {format_shape(vector)}")
    eprint(f"Layer alignment: {layer_alignment_note}; offset={layer_offset}")

    random_sample_rows: List[Dict[str, Any]] = []
    if args.dry_run:
        processed_layers = processed_layer_count(lape_layers, args.max_layers)
        summary: SummaryRow = {
            "status": "dry_run",
            "reason": "Validated inputs without loading model weights.",
            "target_dialect": args.target_dialect,
            "model_id": resolve_model_id(args.model),
            "vector_path": str(vector_path),
            "vector_shape": format_shape(vector),
            "layer_alignment": layer_alignment_note,
            "layer_offset": layer_offset,
            "lape_layers": lape_layers,
            "processed_layers": processed_layers,
            "hidden_dim": int(vector.shape[1]),
            "intermediate_size": intermediate_size,
            "selected_neurons": count_selected_in_processed_layers(selected_by_layer, processed_layers),
        }
        layer_rows: List[LayerRow] = []
    else:
        model = load_model(torch, AutoModelForCausalLM, args)
        summary, layer_rows = measure_projection(
            torch=torch,
            model=model,
            vector=vector,
            selected_by_layer=selected_by_layer,
            lape_layers=lape_layers,
            intermediate_size=intermediate_size,
            layer_offset=layer_offset,
            layer_alignment_note=layer_alignment_note,
            args=args,
        )
        random_stats, random_sample_rows = measure_random_baseline(
            torch=torch,
            model=model,
            vector=vector,
            selected_by_layer=selected_by_layer,
            lape_layers=lape_layers,
            intermediate_size=intermediate_size,
            layer_offset=layer_offset,
            total_energy=float(summary["total_energy"]),
            args=args,
        )
        if random_stats:
            summary.update(random_stats)
            summary["coverage_minus_random_mean"] = float(summary["coverage"]) - float(summary["random_coverage_mean"])
            if float(summary["random_coverage_std"]) > 0:
                summary["coverage_random_z"] = (
                    float(summary["coverage"]) - float(summary["random_coverage_mean"])
                ) / float(summary["random_coverage_std"])
            else:
                summary["coverage_random_z"] = ""

    include_random = args.random_baseline > 0
    write_csv(out_dir / "residual_projection_summary.csv", [summary], summary_fieldnames(include_random))
    write_csv(out_dir / "residual_projection_by_layer.csv", layer_rows, layer_fieldnames())
    if random_sample_rows:
        write_csv(
            out_dir / "residual_projection_random_samples.csv",
            random_sample_rows,
            random_sample_fieldnames(),
        )
    write_json(
        out_dir / "residual_projection_summary.json",
        {
            "neurons_dir": str(neurons_dir),
            "selected_csv": str(selected_csv),
            "vector_path": str(vector_path),
            "out_dir": str(out_dir),
            "target_dialect": args.target_dialect,
            "model_id": resolve_model_id(args.model),
            "random_baseline": args.random_baseline,
            "random_exclude_selected": args.random_exclude_selected,
            "seed": args.seed,
            "summary": summary,
            "layers": layer_rows,
            "random_samples": random_sample_rows,
            "interpretation": (
                "Coverage is residual-subspace projection coverage: the fraction of residual-vector "
                "energy lying in the span of selected neurons' MLP down-projection directions."
            ),
        },
    )
    write_report(out_dir / "REPORT.md", summary, layer_rows, include_random)

    eprint(f"Wrote: {out_dir / 'residual_projection_summary.csv'}")
    eprint(f"Wrote: {out_dir / 'residual_projection_by_layer.csv'}")
    if random_sample_rows:
        eprint(f"Wrote: {out_dir / 'residual_projection_random_samples.csv'}")
    eprint(f"Wrote: {out_dir / 'residual_projection_summary.json'}")
    eprint(f"Wrote: {out_dir / 'REPORT.md'}")
    if summary.get("status") == "ok":
        eprint(f"Residual-subspace coverage: {float(summary['coverage_percent']):.4f}%")


if __name__ == "__main__":
    main()
