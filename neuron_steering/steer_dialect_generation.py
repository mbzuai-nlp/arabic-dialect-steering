#!/usr/bin/env python3
"""
Inference-time dialect steering with LAPE-selected MLP neurons.

This script loads a Hugging Face causal LM, loads dialect-neuron outputs from a
LAPE/RUCAI-style extraction directory, and steers generation by:

  1. amplifying target-dialect neurons by alpha; and
  2. suppressing MSA neurons by gamma; and
  3. optionally suppressing non-target competitor dialect neurons by competitor_gamma.

It supports three timing modes:

  --intervention_mode prefill   : intervene only during the prompt/prefill pass
  --intervention_mode decode    : intervene only during token-by-token decoding
  --intervention_mode both      : intervene during both prefill and decoding

Expected neuron directory contents, in order of preference:

  selected_neurons.csv   with columns such as dialect, layer, neuron, p_CAI, ...
  neurons.pth            RUCAI-style List[List[Tensor]] indexed by dialect/layer
  dialects.json          mapping dialect names to IDs, or IDs to names

The script is intentionally one file and model-family generic. It patches common
MLP forms used by LLaMA/Qwen/Gemma-like gated MLPs and simple non-gated MLPs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_ALIASES = {
    "allam": "humain-ai/ALLaM-7B-Instruct-preview",
    "allam-7b-instruct-preview": "humain-ai/ALLaM-7B-Instruct-preview",
    "fanar": "QCRI/Fanar-1-9B",
    "fanar-1-9b": "QCRI/Fanar-1-9B",
    "fanar-instruct": "QCRI/Fanar-1-9B-Instruct",
    "fanar-1-9b-instruct": "QCRI/Fanar-1-9B-Instruct",
    "jais2": "inceptionai/Jais-2-8B-Chat",
    "jais2-8b-chat": "inceptionai/Jais-2-8B-Chat",
    "qwen3": "Qwen/Qwen3-8B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-8b-instruct": "Qwen/Qwen3-8B",
}


# ----------------------------- basic utilities -----------------------------


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_bool_flag(value: Optional[str]) -> bool:
    if value is None:
        return True
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_model_id(model_or_id: str) -> str:
    return MODEL_ALIASES.get(model_or_id.lower(), model_or_id)


def choose_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_input_device(model: nn.Module) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def flatten_layer_dict(d: Dict[int, Set[int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for layer in sorted(d):
        for neuron in sorted(d[layer]):
            out.append((layer, neuron))
    return out


def count_layer_dict(d: Dict[int, Set[int]]) -> int:
    return sum(len(v) for v in d.values())


def layer_dict_to_jsonable(d: Dict[int, Set[int]]) -> Dict[str, List[int]]:
    return {str(k): sorted(int(x) for x in v) for k, v in sorted(d.items())}


def write_neuron_set_csv(path: str | Path, label: str, role: str, neurons: Dict[int, Set[int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["role", "dialect", "layer", "neuron"])
        writer.writeheader()
        for layer, neuron in flatten_layer_dict(neurons):
            writer.writerow({"role": role, "dialect": label, "layer": layer, "neuron": neuron})


# ----------------------------- dialect metadata -----------------------------


def load_dialects(neurons_dir: str | Path) -> Optional[List[str]]:
    ndir = Path(neurons_dir)
    candidates = [ndir / "dialects.json", ndir / "run_summary.json", ndir / "config_used.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue

        if isinstance(data, list):
            return [str(x) for x in data]

        if isinstance(data, dict):
            if "dialects" in data and isinstance(data["dialects"], list):
                return [str(x) for x in data["dialects"]]

            # Format: {"0": "CAI", "1": "MSA"}
            if all(str(k).isdigit() for k in data.keys()):
                try:
                    return [str(data[str(i)]) for i in range(len(data))]
                except Exception:
                    pass

            # Format: {"CAI": 0, "MSA": 1}
            if all(isinstance(v, int) or str(v).isdigit() for v in data.values()):
                try:
                    pairs = sorted(((int(v), str(k)) for k, v in data.items()), key=lambda x: x[0])
                    return [name for _, name in pairs]
                except Exception:
                    pass
    return None


def dialect_index(dialects: Sequence[str], dialect: str) -> int:
    for i, d in enumerate(dialects):
        if d == dialect:
            return i
    lowered = {d.lower(): i for i, d in enumerate(dialects)}
    if dialect.lower() in lowered:
        return lowered[dialect.lower()]
    raise ValueError(f"Dialect {dialect!r} not found. Available dialects: {list(dialects)}")


# ----------------------------- neuron set loading ---------------------------


@dataclass
class CsvNeuronRows:
    rows: List[Dict[str, str]]
    fieldnames: List[str]
    probability_columns: Dict[str, str]
    selected_dialects_by_pair: Dict[Tuple[int, int], Set[str]]


def detect_probability_columns(fieldnames: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for c in fieldnames:
        if c.startswith("p_") and len(c) > 2:
            mapping[c[2:]] = c
        elif c.startswith("prob_") and len(c) > 5:
            mapping[c[5:]] = c
        elif c.startswith("activation_probability_"):
            mapping[c[len("activation_probability_"):]] = c
    return mapping


def load_selected_neurons_csv(path: str | Path) -> CsvNeuronRows:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = [str(x).strip() for x in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            # Normalize keys in case a file has BOM/whitespace around headers.
            row = {str(k).strip().lstrip("\ufeff"): ("" if v is None else str(v)) for k, v in raw.items()}
            rows.append(row)

    if "dialect" not in fieldnames:
        raise ValueError(f"selected_neurons.csv must contain a 'dialect' column. Columns: {fieldnames}")
    if "layer" not in fieldnames or "neuron" not in fieldnames:
        raise ValueError(f"selected_neurons.csv must contain 'layer' and 'neuron'. Columns: {fieldnames}")

    selected: Dict[Tuple[int, int], Set[str]] = {}
    for row in rows:
        layer = maybe_int(row.get("layer"))
        neuron = maybe_int(row.get("neuron"))
        dialect = row.get("dialect", "").strip()
        if layer is None or neuron is None or dialect == "":
            continue
        selected.setdefault((layer, neuron), set()).add(dialect)

    return CsvNeuronRows(
        rows=rows,
        fieldnames=fieldnames,
        probability_columns=detect_probability_columns(fieldnames),
        selected_dialects_by_pair=selected,
    )


def row_target_metrics(
    row: Dict[str, str],
    dialect: str,
    probability_columns: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    if dialect not in probability_columns:
        return None

    p_target = maybe_float(row.get(probability_columns[dialect]))
    if p_target is None:
        return None

    other_values: List[Tuple[str, float]] = []
    all_values: Dict[str, float] = {}
    for d, col in probability_columns.items():
        p = maybe_float(row.get(col))
        if p is None:
            continue
        all_values[d] = p
        if d != dialect:
            other_values.append((d, p))

    if not other_values:
        return None

    max_other_dialect, max_other = max(other_values, key=lambda x: x[1])
    mean_other = sum(p for _, p in other_values) / max(1, len(other_values))
    winner = max(all_values.items(), key=lambda x: x[1])[0]
    margin = p_target - max_other
    ratio = p_target / (mean_other + 1e-12)
    return {
        "p_target": p_target,
        "max_other": max_other,
        "max_other_dialect": max_other_dialect,
        "mean_other": mean_other,
        "winner": winner,
        "margin": margin,
        "ratio": ratio,
        "all_values": all_values,
    }


def add_to_layer_dict(d: Dict[int, Set[int]], layer: int, neuron: int) -> None:
    d.setdefault(int(layer), set()).add(int(neuron))



def parse_dialect_list(value: Optional[str]) -> List[str]:
    """Parse a comma/space-separated dialect list while preserving order."""
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for part in text.replace(";", ",").replace(" ", ",").split(","):
        item = part.strip()
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def merge_layer_dicts(dicts: Iterable[Dict[int, Set[int]]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for d in dicts:
        for layer, neurons in d.items():
            out.setdefault(int(layer), set()).update(int(n) for n in neurons)
    return out


def subtract_layer_dict(base: Dict[int, Set[int]], remove: Dict[int, Set[int]]) -> int:
    """Remove neurons from base in-place. Return number removed."""
    removed = 0
    for layer, neurons in list(base.items()):
        overlap = neurons & remove.get(layer, set())
        if overlap:
            neurons.difference_update(overlap)
            removed += len(overlap)
        if not neurons:
            del base[layer]
    return removed



def parse_layer_spec(value: Optional[str]) -> Optional[Set[int]]:
    """Parse a layer spec such as "0,1,5-10" into a set of layer IDs.

    Empty strings, "all", "none", and "*" mean no restriction.
    Ranges are inclusive: "5-10" keeps layers 5,6,...,10.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "all", "*", "none", "null"}:
        return None

    layers: Set[int] = set()
    for raw in text.replace(";", ",").replace(" ", ",").split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            bits = [b.strip() for b in part.split("-", 1)]
            if len(bits) != 2 or bits[0] == "" or bits[1] == "":
                raise ValueError(f"Invalid layer range: {part!r}")
            start, end = int(bits[0]), int(bits[1])
            if start > end:
                start, end = end, start
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return layers


def filter_layer_dict_by_layers(
    d: Dict[int, Set[int]],
    allowed_layers: Optional[Set[int]] = None,
    excluded_layers: Optional[Set[int]] = None,
) -> Tuple[Dict[int, Set[int]], Dict[str, Any]]:
    """Return a copy of a layer->neuron dict restricted by include/exclude layers."""
    out: Dict[int, Set[int]] = {}
    removed_neurons = 0
    removed_layers: List[int] = []
    for layer, neurons in sorted(d.items()):
        keep = True
        if allowed_layers is not None and layer not in allowed_layers:
            keep = False
        if excluded_layers is not None and layer in excluded_layers:
            keep = False
        if keep:
            out[int(layer)] = set(int(n) for n in neurons)
        else:
            removed_neurons += len(neurons)
            removed_layers.append(int(layer))
    return out, {
        "input_neurons": count_layer_dict(d),
        "input_layers": len(d),
        "output_neurons": count_layer_dict(out),
        "output_layers": len(out),
        "removed_neurons": removed_neurons,
        "removed_layers": sorted(set(removed_layers)),
        "allowed_layers": None if allowed_layers is None else sorted(allowed_layers),
        "excluded_layers": None if excluded_layers is None else sorted(excluded_layers),
    }

def layer_dict_overlap(a: Dict[int, Set[int]], b: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for layer in sorted(set(a) | set(b)):
        overlap = a.get(layer, set()) & b.get(layer, set())
        if overlap:
            out[layer] = set(overlap)
    return out


def infer_intermediate_size(neurons_dir: str | Path) -> int:
    ndir = Path(neurons_dir)
    for name in ["run_summary.json", "config_used.json", "model_hook_info.json"]:
        path = ndir / name
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ["intermediate_size", "mlp_intermediate_size", "ffn_dim"]:
                value = maybe_int(data.get(key))
                if value is not None and value > 0:
                    return value
    raise ValueError(
        f"Could not infer intermediate_size from {ndir}. "
        "Expected run_summary.json or config_used.json to contain intermediate_size."
    )


def load_all_selected_neurons_by_layer(neurons_dir: str | Path) -> Dict[int, Set[int]]:
    ndir = Path(neurons_dir)
    csv_path = ndir / "selected_neurons.csv"
    if csv_path.exists():
        csv_data = load_selected_neurons_csv(csv_path)
        out: Dict[int, Set[int]] = {}
        for layer, neuron in csv_data.selected_dialects_by_pair:
            add_to_layer_dict(out, layer, neuron)
        return out

    pth_path = ndir / "neurons.pth"
    if not pth_path.exists():
        return {}

    obj = torch.load(pth_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["neurons", "lang_neurons", "data"]:
            if key in obj:
                obj = obj[key]
                break

    out: Dict[int, Set[int]] = {}
    if isinstance(obj, (list, tuple)):
        for dialect_layers in obj:
            if not isinstance(dialect_layers, (list, tuple)):
                continue
            for layer_id, neurons in enumerate(dialect_layers):
                if neurons is None:
                    continue
                if torch.is_tensor(neurons):
                    values = neurons.detach().cpu().view(-1).tolist()
                elif isinstance(neurons, (list, tuple, set)):
                    values = list(neurons)
                else:
                    values = []
                for neuron in values:
                    add_to_layer_dict(out, layer_id, int(neuron))
    return out


def randomize_neuron_roles(
    role_sets: Sequence[Tuple[str, Dict[int, Set[int]]]],
    intermediate_size: int,
    seed: int,
    exclude_selected: bool,
    selected_by_layer: Optional[Dict[int, Set[int]]] = None,
) -> Tuple[List[Dict[int, Set[int]]], Dict[str, Any]]:
    rng = random.Random(seed)
    selected_by_layer = selected_by_layer or {}
    randomized: List[Dict[int, Set[int]]] = []
    used_by_layer: Dict[int, Set[int]] = {}
    role_stats: Dict[str, Any] = {}
    all_layers = sorted({layer for _, role in role_sets for layer in role})

    for role_name, original in role_sets:
        out: Dict[int, Set[int]] = {}
        per_layer_counts: Dict[str, int] = {}
        for layer in all_layers:
            count = len(original.get(layer, set()))
            if count <= 0:
                continue

            blocked = set(used_by_layer.get(layer, set()))
            if exclude_selected:
                blocked.update(selected_by_layer.get(layer, set()))
            available = [n for n in range(intermediate_size) if n not in blocked]
            if count > len(available):
                raise ValueError(
                    f"Cannot sample {count} random {role_name} neurons for layer {layer}; "
                    f"only {len(available)} neurons remain available."
                )

            sampled = set(rng.sample(available, count))
            out[layer] = sampled
            used_by_layer.setdefault(layer, set()).update(sampled)
            per_layer_counts[str(layer)] = count

        randomized.append(out)
        role_stats[role_name] = {
            "original_neurons": count_layer_dict(original),
            "random_neurons": count_layer_dict(out),
            "original_layers": len(original),
            "random_layers": len(out),
            "per_layer_counts": per_layer_counts,
        }

    excluded_counts = {str(layer): len(neurons) for layer, neurons in sorted(selected_by_layer.items())}
    stats = {
        "enabled": True,
        "seed": seed,
        "intermediate_size": intermediate_size,
        "exclude_selected": exclude_selected,
        "role_sampling_order": [name for name, _ in role_sets],
        "roles_are_disjoint": True,
        "role_stats": role_stats,
        "excluded_selected_neurons_by_layer": excluded_counts if exclude_selected else {},
        "excluded_selected_neurons_total": count_layer_dict(selected_by_layer) if exclude_selected else 0,
    }
    return randomized, stats


def filter_csv_neurons_for_dialect(
    csv_data: CsvNeuronRows,
    dialect: str,
    only_highest: bool,
    exclude_shared: bool,
    min_margin: float,
    min_ratio: float,
    max_total: Optional[int],
    max_per_layer: Optional[int],
    role_name: str,
    exclude_if_shared_with: Optional[Set[str]] = None,
    compare_margin_dialect: Optional[str] = None,
    min_margin_over_compare: float = 0.0,
) -> Tuple[Dict[int, Set[int]], Dict[str, Any]]:
    """Load and filter neurons for one dialect from selected_neurons.csv.

    The extra competitor-suppression controls are intentionally generic:
    - exclude_if_shared_with={target} prevents suppressing neurons also selected
      for the target dialect.
    - compare_margin_dialect=target plus min_margin_over_compare keeps only
      neurons whose activation probability is clearly above the target dialect.
    """
    scored: List[Tuple[float, int, int, Dict[str, Any]]] = []
    skipped = {
        "wrong_dialect": 0,
        "bad_layer_or_neuron": 0,
        "shared": 0,
        "shared_with_blocked_dialect": 0,
        "not_highest": 0,
        "low_margin": 0,
        "low_ratio": 0,
        "low_margin_over_compare": 0,
        "missing_probability_columns": 0,
    }

    exclude_if_shared_with = set(exclude_if_shared_with or set())
    pcols_available = dialect in csv_data.probability_columns
    needs_pcols = (
        only_highest
        or min_margin > 0.0
        or min_ratio > 0.0
        or (compare_margin_dialect is not None and min_margin_over_compare > 0.0)
    )
    if needs_pcols and not pcols_available:
        warnings.warn(
            f"{role_name}: probability columns for dialect {dialect!r} are missing, "
            "so highest/margin/ratio filters cannot be applied."
        )

    if compare_margin_dialect is not None and min_margin_over_compare > 0.0:
        if compare_margin_dialect not in csv_data.probability_columns:
            warnings.warn(
                f"{role_name}: probability column for comparison dialect {compare_margin_dialect!r} is missing."
            )

    for row in csv_data.rows:
        if row.get("dialect", "").strip() != dialect:
            skipped["wrong_dialect"] += 1
            continue
        layer = maybe_int(row.get("layer"))
        neuron = maybe_int(row.get("neuron"))
        if layer is None or neuron is None:
            skipped["bad_layer_or_neuron"] += 1
            continue

        pair = (layer, neuron)
        selected_dialects = csv_data.selected_dialects_by_pair.get(pair, set())
        if exclude_shared and len(selected_dialects) > 1:
            skipped["shared"] += 1
            continue
        if exclude_if_shared_with and (selected_dialects & exclude_if_shared_with):
            skipped["shared_with_blocked_dialect"] += 1
            continue

        metrics = row_target_metrics(row, dialect, csv_data.probability_columns)
        if metrics is None:
            if needs_pcols:
                skipped["missing_probability_columns"] += 1
                continue
            # If no p columns, keep the row with a neutral score.
            score = 0.0
        else:
            if only_highest and metrics["winner"] != dialect:
                skipped["not_highest"] += 1
                continue
            if metrics["margin"] < min_margin:
                skipped["low_margin"] += 1
                continue
            if min_ratio > 0.0 and metrics["ratio"] < min_ratio:
                skipped["low_ratio"] += 1
                continue
            if compare_margin_dialect is not None and min_margin_over_compare > 0.0:
                compare_value = metrics["all_values"].get(compare_margin_dialect)
                if compare_value is None:
                    skipped["missing_probability_columns"] += 1
                    continue
                if (float(metrics["p_target"]) - float(compare_value)) < min_margin_over_compare:
                    skipped["low_margin_over_compare"] += 1
                    continue
            # Larger margin first, then higher target activation.
            score = float(metrics["margin"]) + 0.001 * float(metrics["p_target"])

        scored.append((score, layer, neuron, metrics or {}))

    # Sort strongest first, deterministic tie-break by layer/neuron.
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    if max_total is not None and max_total > 0:
        scored = scored[:max_total]

    out: Dict[int, Set[int]] = {}
    per_layer_counts: Dict[int, int] = {}
    for _, layer, neuron, _ in scored:
        if max_per_layer is not None and max_per_layer > 0:
            if per_layer_counts.get(layer, 0) >= max_per_layer:
                continue
        add_to_layer_dict(out, layer, neuron)
        per_layer_counts[layer] = per_layer_counts.get(layer, 0) + 1

    stats = {
        "dialect": dialect,
        "role": role_name,
        "source": "selected_neurons.csv",
        "kept_neurons": count_layer_dict(out),
        "kept_layers": len(out),
        "probability_columns_available": pcols_available,
        "filters": {
            "only_highest": only_highest,
            "exclude_shared": exclude_shared,
            "exclude_if_shared_with": sorted(exclude_if_shared_with),
            "compare_margin_dialect": compare_margin_dialect,
            "min_margin_over_compare": min_margin_over_compare,
            "min_margin": min_margin,
            "min_ratio": min_ratio,
            "max_total": max_total,
            "max_per_layer": max_per_layer,
        },
        "skipped": skipped,
    }
    return out, stats


def load_rucai_neurons_pth(
    neurons_dir: str | Path,
    target_dialect: str,
    msa_dialect: str,
    competitor_dialects: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]], Dict[str, Any]]:
    ndir = Path(neurons_dir)
    path = ndir / "neurons.pth"
    if not path.exists():
        raise FileNotFoundError(f"Could not find neurons.pth in {ndir}")

    dialects = load_dialects(ndir)
    if dialects is None:
        raise FileNotFoundError("Could not infer dialect order. Need dialects.json or run_summary.json.")

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["neurons", "lang_neurons", "data"]:
            if key in obj:
                obj = obj[key]
                break

    if not isinstance(obj, (list, tuple)):
        raise ValueError("neurons.pth must be a RUCAI-style list/tuple indexed by dialect and layer.")

    competitor_dialects = list(competitor_dialects or [])
    target_idx = dialect_index(dialects, target_dialect)
    msa_idx = dialect_index(dialects, msa_dialect) if target_dialect != msa_dialect else -1
    competitor_indices: List[Tuple[str, int]] = []
    for d in competitor_dialects:
        if d.lower() in {target_dialect.lower(), msa_dialect.lower()}:
            continue
        competitor_indices.append((d, dialect_index(dialects, d)))

    def convert(dialect_id: int) -> Dict[int, Set[int]]:
        if dialect_id < 0:
            return {}
        layer_list = obj[dialect_id]
        out: Dict[int, Set[int]] = {}
        for layer_id, neurons in enumerate(layer_list):
            if neurons is None:
                continue
            if torch.is_tensor(neurons):
                vals = neurons.detach().cpu().view(-1).tolist()
            elif isinstance(neurons, (list, tuple, set)):
                vals = list(neurons)
            else:
                vals = []
            for n in vals:
                add_to_layer_dict(out, layer_id, int(n))
        return out

    target = convert(target_idx)
    suppress = convert(msa_idx)
    competitor_sets: Dict[str, Dict[int, Set[int]]] = {d: convert(i) for d, i in competitor_indices}
    competitor = merge_layer_dicts(competitor_sets.values())
    stats = {
        "source": "neurons.pth",
        "dialects": list(dialects),
        "target_dialect": target_dialect,
        "target_index": target_idx,
        "msa_dialect": msa_dialect,
        "msa_index": msa_idx,
        "competitor_dialects": [d for d, _ in competitor_indices],
        "competitor_indices": {d: i for d, i in competitor_indices},
        "target_neurons": count_layer_dict(target),
        "suppress_neurons": count_layer_dict(suppress),
        "competitor_neurons": count_layer_dict(competitor),
        "competitor_neurons_by_dialect": {d: count_layer_dict(v) for d, v in competitor_sets.items()},
        "note": "PTH source cannot apply probability-column filters; use selected_neurons.csv for filtered competitor suppression.",
    }
    return target, suppress, competitor, stats


def resolve_pair_conflicts(
    primary: Dict[int, Set[int]],
    secondary: Dict[int, Set[int]],
    policy: str,
    primary_name: str,
    secondary_name: str,
) -> Dict[str, Any]:
    """Resolve overlaps between two neuron sets in-place."""
    conflicts = layer_dict_overlap(primary, secondary)

    if policy == "drop":
        for layer, overlap in conflicts.items():
            primary.get(layer, set()).difference_update(overlap)
            secondary.get(layer, set()).difference_update(overlap)
    elif policy == "primary_wins":
        for layer, overlap in conflicts.items():
            secondary.get(layer, set()).difference_update(overlap)
    elif policy == "secondary_wins":
        for layer, overlap in conflicts.items():
            primary.get(layer, set()).difference_update(overlap)
    elif policy == "target_wins":
        # Backward-compatible alias when primary is the target set.
        for layer, overlap in conflicts.items():
            secondary.get(layer, set()).difference_update(overlap)
    elif policy == "suppress_wins":
        # Backward-compatible alias when secondary is a suppress set.
        for layer, overlap in conflicts.items():
            primary.get(layer, set()).difference_update(overlap)
    else:
        raise ValueError(f"Unknown conflict policy: {policy}")

    for d in [primary, secondary]:
        for layer in list(d.keys()):
            if not d[layer]:
                del d[layer]

    return {
        "policy": policy,
        "primary_name": primary_name,
        "secondary_name": secondary_name,
        "conflict_neurons": sum(len(v) for v in conflicts.values()),
        "conflict_layers": len(conflicts),
        "conflicts_by_layer": layer_dict_to_jsonable(conflicts),
    }


def resolve_conflicts(
    target: Dict[int, Set[int]],
    suppress: Dict[int, Set[int]],
    policy: str,
) -> Dict[str, Any]:
    # Original API preserved for compatibility.
    return resolve_pair_conflicts(target, suppress, policy, "target", "suppress")


def load_steering_neuron_sets(args: argparse.Namespace) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]], Dict[str, Any]]:
    ndir = Path(args.neurons_dir)
    csv_path = ndir / "selected_neurons.csv"
    pth_path = ndir / "neurons.pth"
    competitor_dialects = parse_dialect_list(args.suppress_competitor_dialects)
    competitor_dialects = [
        d for d in competitor_dialects
        if d.lower() not in {args.target_dialect.lower(), args.msa_dialect.lower()}
    ]

    use_csv = args.neuron_source == "csv" or (args.neuron_source == "auto" and csv_path.exists())
    use_pth = args.neuron_source == "pth" or (args.neuron_source == "auto" and not csv_path.exists() and pth_path.exists())

    competitor: Dict[int, Set[int]] = {}
    competitor_stats: Dict[str, Any] = {"enabled": bool(competitor_dialects), "dialects": competitor_dialects}

    if use_csv:
        csv_data = load_selected_neurons_csv(csv_path)
        if len(competitor_dialects) == 1 and competitor_dialects[0].lower() in {"auto", "all", "all_non_target", "non_target", "competitors"}:
            available = list(csv_data.probability_columns.keys())
            if not available:
                available = sorted({
                    row.get("dialect", "").strip()
                    for row in csv_data.rows
                    if row.get("dialect", "").strip()
                })
            competitor_dialects = [
                d for d in available
                if d.lower() not in {args.target_dialect.lower(), args.msa_dialect.lower()}
            ]
        target, target_stats = filter_csv_neurons_for_dialect(
            csv_data=csv_data,
            dialect=args.target_dialect,
            only_highest=args.only_target_highest,
            exclude_shared=args.exclude_shared,
            min_margin=args.min_margin,
            min_ratio=args.min_ratio,
            max_total=args.max_target_neurons,
            max_per_layer=args.max_neurons_per_layer,
            role_name="target_amplify",
        )
        if args.target_dialect == args.msa_dialect:
            suppress: Dict[int, Set[int]] = {}
            suppress_stats: Dict[str, Any] = {"skipped": "target_dialect equals msa_dialect"}
        else:
            suppress, suppress_stats = filter_csv_neurons_for_dialect(
                csv_data=csv_data,
                dialect=args.msa_dialect,
                only_highest=args.only_msa_highest,
                exclude_shared=args.exclude_shared_msa,
                min_margin=args.msa_min_margin,
                min_ratio=args.msa_min_ratio,
                max_total=args.max_msa_neurons,
                max_per_layer=args.max_neurons_per_layer,
                role_name="msa_suppress",
            )

        competitor_sets: Dict[str, Dict[int, Set[int]]] = {}
        competitor_dialect_stats: Dict[str, Any] = {}
        if competitor_dialects:
            blocked: Set[str] = {args.target_dialect} if args.exclude_target_shared_from_suppression else set()
            for dialect in competitor_dialects:
                dset, dstats = filter_csv_neurons_for_dialect(
                    csv_data=csv_data,
                    dialect=dialect,
                    only_highest=args.only_competitor_highest,
                    exclude_shared=args.exclude_shared_competitors,
                    min_margin=args.competitor_min_margin,
                    min_ratio=args.competitor_min_ratio,
                    max_total=args.max_competitor_neurons_per_dialect,
                    max_per_layer=args.max_neurons_per_layer,
                    role_name=f"competitor_suppress_{dialect}",
                    exclude_if_shared_with=blocked,
                    compare_margin_dialect=args.target_dialect,
                    min_margin_over_compare=args.competitor_min_margin_over_target,
                )
                competitor_sets[dialect] = dset
                competitor_dialect_stats[dialect] = dstats
            competitor = merge_layer_dicts(competitor_sets.values())

            if args.max_competitor_neurons is not None and args.max_competitor_neurons > 0:
                # Global cap after union. Without per-neuron scores across dialects, use deterministic layer/neuron order.
                pairs = flatten_layer_dict(competitor)[: args.max_competitor_neurons]
                capped: Dict[int, Set[int]] = {}
                for layer, neuron in pairs:
                    add_to_layer_dict(capped, layer, neuron)
                competitor = capped

            competitor_stats = {
                "enabled": True,
                "dialects": competitor_dialects,
                "kept_neurons_union_before_conflicts": count_layer_dict(competitor),
                "kept_layers_union_before_conflicts": len(competitor),
                "per_dialect": competitor_dialect_stats,
                "filters": {
                    "only_highest": args.only_competitor_highest,
                    "exclude_shared_competitors": args.exclude_shared_competitors,
                    "exclude_target_shared_from_suppression": args.exclude_target_shared_from_suppression,
                    "competitor_min_margin": args.competitor_min_margin,
                    "competitor_min_ratio": args.competitor_min_ratio,
                    "competitor_min_margin_over_target": args.competitor_min_margin_over_target,
                    "max_competitor_neurons": args.max_competitor_neurons,
                    "max_competitor_neurons_per_dialect": args.max_competitor_neurons_per_dialect,
                    "max_neurons_per_layer": args.max_neurons_per_layer,
                },
            }

        source_stats = {
            "source": "selected_neurons.csv",
            "csv_path": str(csv_path),
            "probability_columns": csv_data.probability_columns,
            "target_stats": target_stats,
            "msa_stats": suppress_stats,
            "competitor_stats": competitor_stats,
        }
    elif use_pth:
        if len(competitor_dialects) == 1 and competitor_dialects[0].lower() in {"auto", "all", "all_non_target", "non_target", "competitors"}:
            dialects_for_auto = load_dialects(ndir)
            if dialects_for_auto is None:
                raise FileNotFoundError("Could not infer dialect order for competitor auto mode. Need dialects.json or run_summary.json.")
            competitor_dialects = [
                d for d in dialects_for_auto
                if d.lower() not in {args.target_dialect.lower(), args.msa_dialect.lower()}
            ]
        target, suppress, competitor, source_stats = load_rucai_neurons_pth(
            ndir, args.target_dialect, args.msa_dialect, competitor_dialects
        )
    else:
        raise FileNotFoundError(
            f"Could not find usable neuron outputs in {ndir}. Expected selected_neurons.csv or neurons.pth."
        )

    target_msa_conflicts = resolve_pair_conflicts(
        target, suppress, args.conflict_policy, "target", "msa_suppress"
    )
    target_competitor_conflicts = resolve_pair_conflicts(
        target, competitor, args.competitor_conflict_policy, "target", "competitor_suppress"
    )

    msa_competitor_overlap = layer_dict_overlap(suppress, competitor)
    if args.msa_wins_over_competitor:
        removed_from_competitor = subtract_layer_dict(competitor, suppress)
    else:
        removed_from_competitor = 0

    # Optional layer-level steering filters. These are applied after dialect/neuron
    # filtering and after conflict resolution so they are easy to interpret:
    # first decide which neurons belong to each role, then decide which layers
    # are allowed to be steered.
    global_layers = parse_layer_spec(args.steer_layers)
    excluded_layers = parse_layer_spec(args.exclude_layers)
    target_layers = parse_layer_spec(args.target_layers) or global_layers
    msa_layers = parse_layer_spec(args.msa_layers) or global_layers
    competitor_layers = parse_layer_spec(args.competitor_layers) or global_layers

    target, target_layer_filter_stats = filter_layer_dict_by_layers(target, target_layers, excluded_layers)
    suppress, suppress_layer_filter_stats = filter_layer_dict_by_layers(suppress, msa_layers, excluded_layers)
    competitor, competitor_layer_filter_stats = filter_layer_dict_by_layers(competitor, competitor_layers, excluded_layers)

    layer_filter_stats = {
        "steer_layers": args.steer_layers,
        "target_layers": args.target_layers,
        "msa_layers": args.msa_layers,
        "competitor_layers": args.competitor_layers,
        "exclude_layers": args.exclude_layers,
        "global_allowed_layers_parsed": None if global_layers is None else sorted(global_layers),
        "excluded_layers_parsed": None if excluded_layers is None else sorted(excluded_layers),
        "target": target_layer_filter_stats,
        "msa_suppress": suppress_layer_filter_stats,
        "competitor_suppress": competitor_layer_filter_stats,
    }

    stats = {
        "target_dialect": args.target_dialect,
        "msa_dialect": args.msa_dialect,
        "competitor_dialects": competitor_dialects,
        "target_neurons_after_conflict_policy": count_layer_dict(target),
        "target_layers_after_conflict_policy": len(target),
        "suppress_neurons_after_conflict_policy": count_layer_dict(suppress),
        "suppress_layers_after_conflict_policy": len(suppress),
        "competitor_neurons_after_conflict_policy": count_layer_dict(competitor),
        "competitor_layers_after_conflict_policy": len(competitor),
        "source_stats": source_stats,
        "layer_filter_stats": layer_filter_stats,
        "conflict_stats": {
            "target_vs_msa": target_msa_conflicts,
            "target_vs_competitor": target_competitor_conflicts,
            "msa_vs_competitor": {
                "msa_wins_over_competitor": args.msa_wins_over_competitor,
                "overlap_neurons": sum(len(v) for v in msa_competitor_overlap.values()),
                "overlap_layers": len(msa_competitor_overlap),
                "removed_from_competitor": removed_from_competitor,
                "overlaps_by_layer": layer_dict_to_jsonable(msa_competitor_overlap),
            },
        },
    }
    return target, suppress, competitor, stats


# ----------------------------- model patching --------------------------------


@dataclass
class SteeringController:
    target_by_layer: Dict[int, Set[int]]
    suppress_by_layer: Dict[int, Set[int]]
    competitor_by_layer: Dict[int, Set[int]]
    alpha: float
    gamma: float
    competitor_gamma: float
    intervention_mode: str
    prefill_positions: str
    decode_positions: str = "last"
    decode_affects_first_token: bool = True
    enabled: bool = True
    current_stage: Optional[str] = None
    _index_cache: Dict[Tuple[str, int, str, int], torch.Tensor] = field(default_factory=dict)
    _warned_invalid: Set[Tuple[str, int, int]] = field(default_factory=set)

    def stage_allowed(self, stage: str) -> bool:
        if self.intervention_mode == "both":
            return stage in {"prefill", "decode"}
        if self.intervention_mode == "decode" and stage == "prefill" and self.decode_affects_first_token:
            # The first generated token is sampled from logits produced by the
            # prompt/prefill pass. This option steers only the last prompt
            # position so decode-mode steering can affect token 1 as well.
            return True
        return stage == self.intervention_mode

    def get_indices(self, role: str, layer_id: int, device: torch.device, dim: int) -> Optional[torch.Tensor]:
        if role == "target":
            source = self.target_by_layer
        elif role == "suppress":
            source = self.suppress_by_layer
        elif role == "competitor":
            source = self.competitor_by_layer
        else:
            raise ValueError(f"Unknown steering role: {role}")
        values = source.get(layer_id)
        if not values:
            return None
        key = (role, layer_id, str(device), dim)
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached

        valid = [int(x) for x in values if 0 <= int(x) < dim]
        invalid_count = len(values) - len(valid)
        if invalid_count > 0 and (role, layer_id, dim) not in self._warned_invalid:
            warnings.warn(
                f"Layer {layer_id} {role}: ignored {invalid_count} neuron indices outside activation dim {dim}."
            )
            self._warned_invalid.add((role, layer_id, dim))
        if not valid:
            return None
        tensor = torch.tensor(sorted(set(valid)), dtype=torch.long, device=device)
        self._index_cache[key] = tensor
        return tensor

    def apply(self, acts: torch.Tensor, layer_id: int) -> torch.Tensor:
        if not self.enabled:
            return acts
        if acts is None or acts.numel() == 0:
            return acts
        if acts.dim() < 2:
            return acts

        # The top-level model pre-hook sets this using past_key_values. If it is
        # missing, use sequence length as a fallback. If the hook reports decode
        # but the activation has multiple positions, prefer the sequence-length
        # evidence and treat it as prefill/chunked-prefill.
        seq_len = acts.shape[-2] if acts.dim() >= 3 else 1
        seq_stage = "decode" if seq_len == 1 else "prefill"
        if self.current_stage is None:
            stage = seq_stage
        elif self.current_stage == "decode" and seq_len > 1:
            stage = "prefill"
        else:
            stage = self.current_stage
        if not self.stage_allowed(stage):
            return acts

        target_idx = self.get_indices("target", layer_id, acts.device, acts.shape[-1])
        suppress_idx = self.get_indices("suppress", layer_id, acts.device, acts.shape[-1])
        competitor_idx = self.get_indices("competitor", layer_id, acts.device, acts.shape[-1])
        if target_idx is None and suppress_idx is None and competitor_idx is None:
            return acts

        # Clone only when we actually modify. This avoids in-place surprises with
        # custom modules while keeping the common path cheap.
        acts = acts.clone()

        def scale_indices(index_tensor: Optional[torch.Tensor], factor: float) -> None:
            if index_tensor is None:
                return
            if acts.dim() == 3:
                decode_first_token_prefill = (
                    stage == "prefill"
                    and self.intervention_mode == "decode"
                    and self.decode_affects_first_token
                )
                use_last = (
                    decode_first_token_prefill
                    or (stage == "prefill" and self.prefill_positions == "last")
                    or (stage == "decode" and self.decode_positions == "last")
                )
                if use_last:
                    acts[:, -1, index_tensor] = acts[:, -1, index_tensor] * factor
                else:
                    acts[:, :, index_tensor] = acts[:, :, index_tensor] * factor
            elif acts.dim() == 2:
                # Shape [tokens, hidden] or [batch, hidden]. There is no explicit
                # sequence axis, so scale all rows.
                acts[:, index_tensor] = acts[:, index_tensor] * factor
            else:
                # Generic fallback: scale along the last dimension.
                slicer = [slice(None)] * acts.dim()
                slicer[-1] = index_tensor
                acts[tuple(slicer)] = acts[tuple(slicer)] * factor

        scale_indices(target_idx, self.alpha)
        scale_indices(suppress_idx, self.gamma)
        scale_indices(competitor_idx, self.competitor_gamma)
        return acts


def find_transformer_layers(model: nn.Module) -> Tuple[str, Sequence[nn.Module]]:
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None:
        n_layers = getattr(model.config, "n_layer", None)

    # Common fast paths.
    candidates = [
        "model.layers",
        "base_model.model.layers",
        "transformer.h",
        "transformer.blocks",
        "gpt_neox.layers",
    ]
    for name in candidates:
        obj: Any = model
        ok = True
        for part in name.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
            return name, obj

    # Robust fallback: find a ModuleList with the expected length whose elements
    # expose an MLP-like submodule.
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            if n_layers is not None and len(module) != int(n_layers):
                continue
            first = module[0]
            if any(hasattr(first, attr) for attr in ["mlp", "feed_forward", "ffn"]):
                return name, module

    raise RuntimeError("Could not locate transformer layers. Try editing find_transformer_layers for this model.")


def get_mlp(layer: nn.Module) -> nn.Module:
    for attr in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise RuntimeError(f"Layer {layer.__class__.__name__} does not expose mlp/feed_forward/ffn.")


def activation_from_name(name: Any) -> Optional[Any]:
    if name is None:
        return None
    key = str(name).lower()
    try:
        from transformers.activations import ACT2FN  # type: ignore
        if key in ACT2FN:
            return ACT2FN[key]
    except Exception:
        pass
    if key in {"silu", "swish"}:
        return F.silu
    if key == "gelu":
        return F.gelu
    if key in {"gelu_new", "gelu_fast", "gelu_pytorch_tanh"}:
        return lambda x: F.gelu(x, approximate="tanh")
    if key == "relu":
        return F.relu
    if key in {"relu2", "squared_relu", "relu_squared"}:
        return lambda x: torch.square(F.relu(x))
    return None


def get_activation_fn(mlp: nn.Module, model_config: Optional[Any] = None) -> Optional[Any]:
    for attr in ["act_fn", "activation_fn", "activation", "act", "gelu", "swiglu"]:
        if hasattr(mlp, attr):
            value = getattr(mlp, attr)
            if callable(value):
                return value
            fn = activation_from_name(value)
            if fn is not None:
                return fn
    if model_config is not None:
        for attr in ["hidden_act", "activation_function", "activation", "act_fn"]:
            fn = activation_from_name(getattr(model_config, attr, None))
            if fn is not None:
                return fn
    return None


def infer_mlp_style(mlp: nn.Module) -> str:
    if all(hasattr(mlp, x) for x in ["gate_proj", "up_proj", "down_proj"]):
        return "gated_gate_up_down"
    if all(hasattr(mlp, x) for x in ["up_proj", "down_proj"]):
        return "nongated_up_down"
    if all(hasattr(mlp, x) for x in ["c_fc", "c_proj"]):
        return "gpt_c_fc"
    if all(hasattr(mlp, x) for x in ["dense_h_to_4h", "dense_4h_to_h"]):
        return "bloom_dense"
    if all(hasattr(mlp, x) for x in ["fc1", "fc2"]):
        return "fc1_fc2"
    return "unknown"


def install_steering_patches(
    model: nn.Module,
    controller: SteeringController,
) -> Tuple[List[Any], List[Tuple[nn.Module, Any]], Dict[str, Any]]:
    layers_name, layers = find_transformer_layers(model)
    original_forwards: List[Tuple[nn.Module, Any]] = []
    patched_layers: List[Dict[str, Any]] = []

    for layer_id, layer in enumerate(layers):
        if (
            layer_id not in controller.target_by_layer
            and layer_id not in controller.suppress_by_layer
            and layer_id not in controller.competitor_by_layer
        ):
            continue
        mlp = get_mlp(layer)
        style = infer_mlp_style(mlp)
        act_fn = get_activation_fn(mlp, getattr(model, "config", None))
        old_forward = mlp.forward

        if style == "gated_gate_up_down":
            if act_fn is None:
                raise RuntimeError(f"Layer {layer_id}: gated MLP has no exposed activation function.")

            def make_forward(m: nn.Module, lid: int, activation_fn: Any):
                def new_forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                    gate = activation_fn(m.gate_proj(hidden_states))
                    gate = controller.apply(gate, lid)
                    up = m.up_proj(hidden_states)
                    return m.down_proj(gate * up)
                return new_forward

            mlp.forward = make_forward(mlp, layer_id, act_fn)  # type: ignore[method-assign]

        elif style == "nongated_up_down":
            if act_fn is None:
                warnings.warn(
                    f"Layer {layer_id}: non-gated up_proj/down_proj MLP has no exposed activation function. "
                    "The patch will scale raw up_proj outputs, which may not match post-activation neurons."
                )

            def make_forward(m: nn.Module, lid: int, activation_fn: Optional[Any]):
                def new_forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                    hidden = m.up_proj(hidden_states)
                    if activation_fn is not None:
                        hidden = activation_fn(hidden)
                    hidden = controller.apply(hidden, lid)
                    if hasattr(m, "dropout") and callable(getattr(m, "dropout")):
                        hidden = m.dropout(hidden)
                    return m.down_proj(hidden)
                return new_forward

            mlp.forward = make_forward(mlp, layer_id, act_fn)  # type: ignore[method-assign]

        elif style == "gpt_c_fc":
            if act_fn is None:
                warnings.warn(
                    f"Layer {layer_id}: c_fc/c_proj MLP has no exposed activation function; scaling raw c_fc outputs."
                )

            def make_forward(m: nn.Module, lid: int, activation_fn: Optional[Any]):
                def new_forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                    hidden = m.c_fc(hidden_states)
                    if activation_fn is not None:
                        hidden = activation_fn(hidden)
                    hidden = controller.apply(hidden, lid)
                    if hasattr(m, "dropout") and callable(getattr(m, "dropout")):
                        hidden = m.dropout(hidden)
                    return m.c_proj(hidden)
                return new_forward

            mlp.forward = make_forward(mlp, layer_id, act_fn)  # type: ignore[method-assign]

        elif style == "bloom_dense":
            if act_fn is None:
                warnings.warn(
                    f"Layer {layer_id}: dense_h_to_4h/dense_4h_to_h MLP has no exposed activation function; "
                    "scaling raw dense_h_to_4h outputs."
                )

            def make_forward(m: nn.Module, lid: int, activation_fn: Optional[Any]):
                def new_forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                    hidden = m.dense_h_to_4h(hidden_states)
                    if activation_fn is not None:
                        hidden = activation_fn(hidden)
                    hidden = controller.apply(hidden, lid)
                    return m.dense_4h_to_h(hidden)
                return new_forward

            mlp.forward = make_forward(mlp, layer_id, act_fn)  # type: ignore[method-assign]

        elif style == "fc1_fc2":
            if act_fn is None:
                warnings.warn(f"Layer {layer_id}: fc1/fc2 MLP has no exposed activation function; scaling raw fc1 outputs.")

            def make_forward(m: nn.Module, lid: int, activation_fn: Optional[Any]):
                def new_forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                    hidden = m.fc1(hidden_states)
                    if activation_fn is not None:
                        hidden = activation_fn(hidden)
                    hidden = controller.apply(hidden, lid)
                    if hasattr(m, "dropout") and callable(getattr(m, "dropout")):
                        hidden = m.dropout(hidden)
                    return m.fc2(hidden)
                return new_forward

            mlp.forward = make_forward(mlp, layer_id, act_fn)  # type: ignore[method-assign]

        else:
            raise RuntimeError(
                f"Layer {layer_id}: unsupported MLP style for module {mlp.__class__.__name__}. "
                "Expected gate_proj/up_proj/down_proj or a common non-gated MLP."
            )

        original_forwards.append((mlp, old_forward))
        patched_layers.append(
            {
                "layer": layer_id,
                "mlp_class": mlp.__class__.__name__,
                "style": style,
                "target_neurons": len(controller.target_by_layer.get(layer_id, set())),
                "suppress_neurons": len(controller.suppress_by_layer.get(layer_id, set())),
                "competitor_suppress_neurons": len(controller.competitor_by_layer.get(layer_id, set())),
            }
        )

    # Top-level pre-hook: detect whether a forward call is prefill or decode.
    def pre_forward_hook(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        past = kwargs.get("past_key_values", None)
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args and torch.is_tensor(args[0]):
            input_ids = args[0]
        seq_len = int(input_ids.shape[1]) if torch.is_tensor(input_ids) and input_ids.dim() >= 2 else None
        # Prefer sequence length when available: prompt/prefill usually has len > 1,
        # while cached decoding usually has len == 1. This also handles HF versions
        # that pass an initialized Cache object even on the first call.
        if seq_len is not None and seq_len > 1:
            controller.current_stage = "prefill"
        elif past is None:
            controller.current_stage = "prefill"
        else:
            controller.current_stage = "decode"

    handles: List[Any] = []
    try:
        handles.append(model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True))
    except TypeError:
        # Old PyTorch fallback: cannot see kwargs, so use seq length from args if possible.
        def old_pre_forward_hook(module: nn.Module, args: Tuple[Any, ...]) -> None:
            if args and torch.is_tensor(args[0]) and args[0].dim() >= 2:
                controller.current_stage = "decode" if args[0].shape[1] == 1 else "prefill"
            else:
                controller.current_stage = None

        handles.append(model.register_forward_pre_hook(old_pre_forward_hook))

    info = {
        "layers_container": layers_name,
        "num_layers_detected": len(layers),
        "patched_layers": patched_layers,
    }
    return handles, original_forwards, info


def restore_patches(handles: List[Any], original_forwards: List[Tuple[nn.Module, Any]]) -> None:
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass
    for module, old_forward in original_forwards:
        module.forward = old_forward  # type: ignore[method-assign]


# ----------------------------- prompt handling ------------------------------


def load_prompts(args: argparse.Namespace) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    if args.prompt is not None:
        prompts.append({"id": "0", "prompt": args.prompt})

    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                obj: Dict[str, Any]
                if line.lstrip().startswith("{"):
                    obj = json.loads(line)
                    prompt = obj.get(args.prompt_field)
                    if prompt is None:
                        raise ValueError(f"Line {i+1} missing prompt field {args.prompt_field!r}")
                    pid = str(obj.get("id", i))
                    prompts.append({"id": pid, "prompt": str(prompt)})
                else:
                    prompts.append({"id": str(i), "prompt": line})

    if not prompts:
        raise ValueError("Provide --prompt or --prompt_file.")
    return prompts


def build_model_input_text(tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
    mode = args.chat_template
    if mode == "never":
        return prompt

    has_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
    if mode == "auto" and not has_template:
        return prompt
    if mode == "always" and not has_template:
        raise ValueError("--chat_template always was requested, but tokenizer has no chat template.")

    messages: List[Dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.chat_template_kwargs:
        extra = json.loads(args.chat_template_kwargs)
        if not isinstance(extra, dict):
            raise ValueError("--chat_template_kwargs must be a JSON object.")
        kwargs.update(extra)

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Some tokenizers do not accept model-specific kwargs such as enable_thinking.
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        return tokenizer.apply_chat_template(messages, **kwargs)


def prepare_tokenizer(model_id: str, args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.slow_tokenizer,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


# ----------------------------- generation -----------------------------------


def generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if args.do_sample:
        kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
        if args.top_k is not None and args.top_k >= 0:
            kwargs["top_k"] = args.top_k
    return kwargs


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    args: argparse.Namespace,
    controller: SteeringController,
    steering_enabled: bool,
) -> Dict[str, Any]:
    input_text = build_model_input_text(tokenizer, prompt, args)
    device = get_input_device(model)
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])

    controller.enabled = steering_enabled
    controller.current_stage = None

    with torch.inference_mode():
        out = model.generate(**inputs, **generation_kwargs(args, tokenizer))

    new_tokens = out[0, input_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {
        "input_text": input_text,
        "input_tokens": input_len,
        "generated_text": generated_text,
        "full_text": full_text,
        "new_tokens": int(new_tokens.numel()),
    }


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    args: argparse.Namespace,
    controller: SteeringController,
    steering_enabled: bool,
) -> List[Dict[str, Any]]:
    if len(prompts) == 1:
        return [generate_one(model, tokenizer, prompts[0], args, controller, steering_enabled)]

    input_texts = [build_model_input_text(tokenizer, prompt, args) for prompt in prompts]
    device = get_input_device(model)
    inputs = tokenizer(input_texts, return_tensors="pt", add_special_tokens=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_width = int(inputs["input_ids"].shape[1])
    attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
    input_token_counts = attention_mask.sum(dim=1).detach().cpu().tolist()

    controller.enabled = steering_enabled
    controller.current_stage = None

    with torch.inference_mode():
        out = model.generate(**inputs, **generation_kwargs(args, tokenizer))

    results: List[Dict[str, Any]] = []
    for row, input_text, input_tokens in zip(out, input_texts, input_token_counts):
        new_tokens = row[input_width:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        full_text = tokenizer.decode(row, skip_special_tokens=True)
        if tokenizer.pad_token_id is None:
            new_token_count = int(new_tokens.numel())
        else:
            new_token_count = int(new_tokens.ne(tokenizer.pad_token_id).sum().item())
        results.append(
            {
                "input_text": input_text,
                "input_tokens": int(input_tokens),
                "generated_text": generated_text,
                "full_text": full_text,
                "new_tokens": new_token_count,
            }
        )
    return results


def load_model(model_id: str, args: argparse.Namespace) -> Any:
    from transformers import AutoModelForCausalLM

    dtype = choose_dtype(args.dtype)
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if args.device_map == "none":
        device = torch.device(args.device)
        model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model


# ----------------------------- CLI ------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate with target dialect neuron amplification, MSA suppression, and optional competitor-dialect suppression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and data.
    p.add_argument("--model", required=True, help="Model alias or Hugging Face model ID.")
    p.add_argument("--neurons_dir", required=True, help="Directory containing selected_neurons.csv or neurons.pth.")
    p.add_argument("--target_dialect", required=True, help="Dialect to amplify, e.g. CAI, BEI, DOH, RAB.")
    p.add_argument("--msa_dialect", default="MSA", help="Dialect label whose neurons are suppressed.")
    p.add_argument("--out_file", required=True, help="Output JSONL file for generations.")
    p.add_argument("--prompt", default=None, help="Single prompt to generate from.")
    p.add_argument("--prompt_file", default=None, help="JSONL file with prompts, or plain text one prompt per line.")
    p.add_argument("--prompt_field", default="prompt", help="Prompt field name in JSONL prompt file.")
    p.add_argument("--batch_size", type=int, default=1, help="Number of prompts per model.generate call.")

    # Loading settings.
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    p.add_argument("--device_map", default="auto", help="HF device_map. Use 'none' to place on --device manually.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--trust_remote_code", type=parse_bool_flag, nargs="?", const=True, default=True)
    p.add_argument("--slow_tokenizer", action="store_true", help="Use slow tokenizer implementation.")

    # Neuron filtering.
    p.add_argument("--neuron_source", choices=["auto", "csv", "pth"], default="auto")
    p.add_argument("--only_target_highest", action="store_true", help="Keep target neurons only if target has the highest p_* value.")
    p.add_argument("--only_msa_highest", action="store_true", help="Keep MSA suppress neurons only if MSA has the highest p_* value.")
    p.add_argument("--exclude_shared", action="store_true", help="Exclude target neurons selected for more than one dialect.")
    p.add_argument("--exclude_shared_msa", action="store_true", help="Exclude MSA suppress neurons selected for more than one dialect.")
    p.add_argument("--min_margin", type=float, default=0.0, help="Target p_d - max(other p) minimum for target neurons.")
    p.add_argument("--min_ratio", type=float, default=0.0, help="Target p_d / mean(other p) minimum for target neurons. 0 disables.")
    p.add_argument("--msa_min_margin", type=float, default=0.0, help="MSA p_d - max(other p) minimum for suppress neurons.")
    p.add_argument("--msa_min_ratio", type=float, default=0.0, help="MSA p_d / mean(other p) minimum for suppress neurons. 0 disables.")
    p.add_argument("--max_target_neurons", type=int, default=None, help="Keep at most this many target neurons after filtering.")
    p.add_argument("--max_msa_neurons", type=int, default=None, help="Keep at most this many MSA suppress neurons after filtering.")
    p.add_argument("--suppress_competitor_dialects", default="", help="Comma-separated non-target dialects to suppress, e.g. BEI,DOH,RAB.")
    p.add_argument("--only_competitor_highest", action="store_true", help="Keep competitor suppress neurons only if that competitor dialect has the highest p_* value.")
    p.add_argument("--exclude_shared_competitors", action="store_true", help="Exclude competitor suppress neurons selected for more than one dialect. Usually leave false to allow BEI+DOH shared competitor neurons.")
    p.add_argument("--exclude_target_shared_from_suppression", action="store_true", default=True, help="Do not suppress competitor neurons that are also selected for the target dialect.")
    p.add_argument("--allow_target_shared_suppression", action="store_false", dest="exclude_target_shared_from_suppression", help="Allow suppression of competitor neurons even if they are also selected for the target dialect. Risky.")
    p.add_argument("--competitor_min_margin", type=float, default=0.0, help="Competitor p_d - max(other p) minimum for competitor suppress neurons.")
    p.add_argument("--competitor_min_ratio", type=float, default=0.0, help="Competitor p_d / mean(other p) minimum for competitor suppress neurons. 0 disables.")
    p.add_argument("--competitor_min_margin_over_target", type=float, default=0.0, help="Keep competitor neurons only if p_competitor - p_target is at least this value.")
    p.add_argument("--max_competitor_neurons", type=int, default=None, help="Global cap for union of all competitor suppress neurons after filtering.")
    p.add_argument("--max_competitor_neurons_per_dialect", type=int, default=None, help="Cap competitor suppress neurons per competitor dialect before union.")
    p.add_argument("--max_neurons_per_layer", type=int, default=None, help="Cap neurons per layer for each set.")
    p.add_argument("--steer_layers", default="", help="Restrict all steering roles to these layers, e.g. '0,1,16-31'. Empty/all means no layer restriction.")
    p.add_argument("--target_layers", default="", help="Restrict target amplification to these layers. Overrides --steer_layers for target neurons.")
    p.add_argument("--msa_layers", default="", help="Restrict MSA suppression to these layers. Overrides --steer_layers for MSA neurons.")
    p.add_argument("--competitor_layers", default="", help="Restrict competitor suppression to these layers. Overrides --steer_layers for competitor neurons.")
    p.add_argument("--exclude_layers", default="", help="Exclude these layers from all steering roles, e.g. '0,31'. Applied after include layer filters.")
    p.add_argument("--conflict_policy", choices=["drop", "target_wins", "suppress_wins"], default="drop", help="How to resolve overlaps between target-amplify and MSA-suppress sets.")
    p.add_argument("--competitor_conflict_policy", choices=["drop", "target_wins", "suppress_wins"], default="target_wins", help="How to resolve overlaps between target-amplify and competitor-suppress sets.")
    p.add_argument("--msa_wins_over_competitor", action="store_true", default=True, help="If a neuron is both MSA-suppress and competitor-suppress, use the MSA gamma only.")
    p.add_argument("--no_msa_wins_over_competitor", action="store_false", dest="msa_wins_over_competitor")
    p.add_argument("--randomize_neurons", action="store_true", help="Replace loaded target/MSA/competitor neuron sets with same-layer, same-count random neurons.")
    p.add_argument("--randomize_exclude_selected", action="store_true", help="When randomizing neurons, sample only from neurons not selected by LAPE for any dialect.")
    p.add_argument("--random_neuron_seed", type=int, default=None, help="Seed for random neuron ablation. Defaults to --seed.")

    # Steering.
    p.add_argument("--alpha", type=float, default=1.3, help="Multiplier for target dialect neurons.")
    p.add_argument("--gamma", type=float, default=0.7, help="Multiplier for MSA neurons. Use 0 for hard suppression.")
    p.add_argument("--competitor_gamma", type=float, default=0.9, help="Multiplier for non-target competitor dialect neurons. Use mild values such as 0.85-0.95.")
    p.add_argument("--intervention_mode", choices=["prefill", "decode", "both"], default="decode")
    p.add_argument("--prefill_positions", choices=["last", "all"], default="last", help="When intervening during prefill/both mode, modify only the last prompt position or all prompt positions.")
    p.add_argument("--decode_positions", choices=["last", "all"], default="last", help="When intervening during decoding, modify only the last position or all positions. With KV cache, decode usually has one position.")
    p.add_argument("--decode_affects_first_token", action="store_true", default=True, help="In decode mode, also steer the last prompt position so the first generated token is affected.")
    p.add_argument("--no_decode_affects_first_token", action="store_false", dest="decode_affects_first_token")
    p.add_argument("--dry_run", action="store_true", help="Load neuron sets and write stats, but do not load model or generate.")

    # Chat formatting.
    p.add_argument("--chat_template", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--chat_template_kwargs", default=None, help="Extra JSON kwargs for apply_chat_template, e.g. '{\"enable_thinking\": false}'.")

    # Generation.
    p.add_argument("--generate_baseline", action="store_true", help="Also generate a no-intervention baseline for each prompt.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    set_seed(args.seed)
    model_id = resolve_model_id(args.model)
    out_file = Path(args.out_file)
    ensure_dir(out_file.parent)


    target_neurons, suppress_neurons, competitor_neurons, neuron_stats = load_steering_neuron_sets(args)
    if args.randomize_neurons:
        random_seed = args.seed if args.random_neuron_seed is None else args.random_neuron_seed
        intermediate_size = infer_intermediate_size(args.neurons_dir)
        excluded_neurons = (
            load_all_selected_neurons_by_layer(args.neurons_dir)
            if args.randomize_exclude_selected
            else {}
        )
        randomized_sets, random_stats = randomize_neuron_roles(
            role_sets=[
                ("target_amplify", target_neurons),
                ("msa_suppress", suppress_neurons),
                ("competitor_suppress", competitor_neurons),
            ],
            intermediate_size=intermediate_size,
            seed=random_seed,
            exclude_selected=args.randomize_exclude_selected,
            selected_by_layer=excluded_neurons,
        )
        target_neurons, suppress_neurons, competitor_neurons = randomized_sets
        neuron_stats["random_neuron_ablation"] = random_stats
        neuron_stats["target_neurons_after_conflict_policy"] = count_layer_dict(target_neurons)
        neuron_stats["target_layers_after_conflict_policy"] = len(target_neurons)
        neuron_stats["suppress_neurons_after_conflict_policy"] = count_layer_dict(suppress_neurons)
        neuron_stats["suppress_layers_after_conflict_policy"] = len(suppress_neurons)
        neuron_stats["competitor_neurons_after_conflict_policy"] = count_layer_dict(competitor_neurons)
        neuron_stats["competitor_layers_after_conflict_policy"] = len(competitor_neurons)

    competitor_dialects = list(neuron_stats.get("competitor_dialects", parse_dialect_list(args.suppress_competitor_dialects)))
    eprint(
        f"Loaded neurons: target {args.target_dialect}={count_layer_dict(target_neurons)} "
        f"across {len(target_neurons)} layers; suppress {args.msa_dialect}={count_layer_dict(suppress_neurons)} "
        f"across {len(suppress_neurons)} layers; competitors {competitor_dialects}={count_layer_dict(competitor_neurons)} "
        f"across {len(competitor_neurons)} layers."
    )

    stats_dir = ensure_dir(out_file.parent / (out_file.stem + "_steering_stats"))
    write_json(stats_dir / "steering_neuron_stats.json", neuron_stats)
    write_json(stats_dir / "target_neurons_by_layer.json", layer_dict_to_jsonable(target_neurons))
    write_json(stats_dir / "msa_suppress_neurons_by_layer.json", layer_dict_to_jsonable(suppress_neurons))
    write_json(stats_dir / "competitor_suppress_neurons_by_layer.json", layer_dict_to_jsonable(competitor_neurons))
    write_neuron_set_csv(stats_dir / "used_target_neurons.csv", args.target_dialect, "target_amplify", target_neurons)
    write_neuron_set_csv(stats_dir / "used_msa_suppress_neurons.csv", args.msa_dialect, "msa_suppress", suppress_neurons)
    write_neuron_set_csv(stats_dir / "used_competitor_suppress_neurons.csv", ",".join(competitor_dialects), "competitor_suppress", competitor_neurons)

    config = vars(args).copy()
    config["resolved_model_id"] = model_id
    write_json(stats_dir / "steering_config.json", config)

    if args.dry_run:
        eprint(f"Dry run complete. Stats written to {stats_dir}")
        return

    if count_layer_dict(target_neurons) == 0:
        warnings.warn("No target neurons after filtering; steering will only apply MSA suppression.")
    if count_layer_dict(suppress_neurons) == 0:
        warnings.warn("No MSA suppress neurons after filtering; steering will only apply target amplification and optional competitor suppression.")
    if competitor_dialects and count_layer_dict(competitor_neurons) == 0:
        warnings.warn("Competitor dialect suppression was requested, but no competitor neurons survived filtering.")

    tokenizer = prepare_tokenizer(model_id, args)
    if args.batch_size > 1 and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    if args.batch_size > 1 and tokenizer.pad_token_id is None:
        raise ValueError("Batch generation requires tokenizer.pad_token_id to be set.")
    model = load_model(model_id, args)

    controller = SteeringController(
        target_by_layer=target_neurons,
        suppress_by_layer=suppress_neurons,
        competitor_by_layer=competitor_neurons,
        alpha=args.alpha,
        gamma=args.gamma,
        competitor_gamma=args.competitor_gamma,
        intervention_mode=args.intervention_mode,
        prefill_positions=args.prefill_positions,
        decode_positions=args.decode_positions,
        decode_affects_first_token=args.decode_affects_first_token,
    )
    handles, original_forwards, patch_info = install_steering_patches(model, controller)
    write_json(stats_dir / "patch_info.json", patch_info)
    eprint(f"Patched {len(patch_info['patched_layers'])} MLP layers. Generation starting.")

    prompts = load_prompts(args)
    gen_count = 0
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            for start in range(0, len(prompts), args.batch_size):
                batch_items = prompts[start : start + args.batch_size]
                batch_prompts = [item["prompt"] for item in batch_items]

                baseline_results: List[Optional[Dict[str, Any]]]
                if args.generate_baseline:
                    baseline_results = generate_batch(
                        model, tokenizer, batch_prompts, args, controller, steering_enabled=False
                    )
                else:
                    baseline_results = [None] * len(batch_items)

                steered_results = generate_batch(
                    model, tokenizer, batch_prompts, args, controller, steering_enabled=True
                )

                for item, baseline, steered in zip(batch_items, baseline_results, steered_results):
                    pid = item["id"]
                    prompt = item["prompt"]
                    record: Dict[str, Any] = {
                        "id": pid,
                        "prompt": prompt,
                        "model_id": model_id,
                        "target_dialect": args.target_dialect,
                        "msa_dialect": args.msa_dialect,
                        "alpha": args.alpha,
                        "gamma": args.gamma,
                        "competitor_gamma": args.competitor_gamma,
                        "competitor_dialects": competitor_dialects,
                        "intervention_mode": args.intervention_mode,
                        "prefill_positions": args.prefill_positions,
                        "decode_positions": args.decode_positions,
                        "decode_affects_first_token": args.decode_affects_first_token,
                    }

                    if baseline is not None:
                        record["baseline_output"] = baseline["generated_text"]
                        record["baseline_full_text"] = baseline["full_text"]

                    record["steered_output"] = steered["generated_text"]
                    record["steered_full_text"] = steered["full_text"]
                    record["input_tokens"] = steered["input_tokens"]
                    record["steered_new_tokens"] = steered["new_tokens"]

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    gen_count += 1
                    eprint(f"Wrote generation {gen_count}/{len(prompts)}: id={pid}")
                f.flush()
    finally:
        restore_patches(handles, original_forwards)

    write_json(
        stats_dir / "run_summary.json",
        {
            "out_file": str(out_file),
            "generations": gen_count,
            "model_id": model_id,
            "target_dialect": args.target_dialect,
            "msa_dialect": args.msa_dialect,
            "target_neurons": count_layer_dict(target_neurons),
            "suppress_neurons": count_layer_dict(suppress_neurons),
            "competitor_neurons": count_layer_dict(competitor_neurons),
            "competitor_dialects": competitor_dialects,
            "intervention_mode": args.intervention_mode,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "competitor_gamma": args.competitor_gamma,
            "decode_positions": args.decode_positions,
            "decode_affects_first_token": args.decode_affects_first_token,
        },
    )
    eprint(f"Done. Generations written to {out_file}")


if __name__ == "__main__":
    main()
