#!/usr/bin/env python3
"""
LAPE dialect-neuron extraction, adapted to the methodology of:
RUCAIBox/Language-Specific-Neurons

This is a one-file Transformers implementation that keeps the RUCAIBox
selection logic while changing two practical parts:
  1. model support: ALLaM, Fanar, Jais2, Qwen3, or any compatible HF CausalLM;
  2. data loading: JSONL manifest -> parallel TSV files aligned by sentID.

Main output compatible with the RUCAIBox format:
  neurons.pth = List[List[LongTensor]]
  neurons[dialect_id][layer_id] -> LongTensor of selected neuron indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.activations import ACT2FN
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires transformers. Install with: "
        "pip install torch transformers accelerate safetensors tqdm"
    ) from exc


MODEL_ALIASES = {
    # User-requested aliases.
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
    "qwen3": "Qwen/Qwen3-8B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-8b-instruct": "Qwen/Qwen3-8B",
    "qwen/qwen3-8b": "Qwen/Qwen3-8B",
}


@dataclass
class ManifestEntry:
    dialect: str
    path: str
    lang: Optional[str] = None


@dataclass
class TsvStats:
    dialect: str
    path: str
    raw_rows: int
    kept_rows: int
    empty_text_rows: int
    duplicate_sentids: int
    unique_sentids: int


@dataclass
class HookSpec:
    layer_id: int
    projection_name: str
    projection_module: nn.Module
    activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
    intermediate_size: int


class ActivationState:
    """Mutable hook state shared during one forward pass."""

    def __init__(self, active_counts: torch.Tensor, positive_threshold: float):
        self.active_counts = active_counts
        self.positive_threshold = positive_threshold
        self.dialect_id: Optional[int] = None
        self.valid_mask: Optional[torch.Tensor] = None

    def reset_batch(self) -> None:
        self.dialect_id = None
        self.valid_mask = None


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def resolve_model_id(model_arg: str) -> str:
    return MODEL_ALIASES.get(model_arg.lower(), model_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract dialect-specific neurons with the RUCAIBox LAPE methodology, "
            "using a parallel TSV JSONL manifest."
        )
    )
    parser.add_argument("--model", required=True, help="Model alias or Hugging Face model ID.")
    parser.add_argument("--data_manifest", required=True, help="JSONL file mapping dialects to TSV files.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")

    # TSV/data controls.
    parser.add_argument("--id_col", default="sentID.BTEC", help="TSV sentence ID column.")
    parser.add_argument("--text_col", default="sent", help="TSV sentence text column.")
    parser.add_argument("--lang_col", default="lang", help="TSV language/dialect column.")
    parser.add_argument("--split_col", default="split", help="TSV split column.")
    parser.add_argument("--split_regex", default=None, help="Optional regex to filter the split column before sentID intersection.")
    parser.add_argument("--no_lang_filter", action="store_true", help="Do not filter rows by the manifest entry's lang field.")
    parser.add_argument(
        "--duplicate_policy",
        choices=["error", "first", "last"],
        default="error",
        help="How to handle duplicate sentIDs within one dialect TSV.",
    )
    parser.add_argument("--max_parallel_sentences", type=int, default=None, help="Limit shared sentIDs after intersection.")
    parser.add_argument(
        "--sentid_order",
        choices=["numeric", "lex", "manifest", "shuffle"],
        default="numeric",
        help=(
            "Order for shared sentIDs. 'manifest' keeps the first dialect TSV order; "
            "'shuffle' uses --seed."
        ),
    )
    parser.add_argument("--seed", type=int, default=13, help="Seed used only for --sentid_order shuffle.")

    # Model/runtime controls.
    parser.add_argument("--seq_len", type=int, default=512, help="Tokenizer max_length for each sentence batch.")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of sentences per model forward pass.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="dtype passed to from_pretrained.",
    )
    parser.add_argument("--device_map", default="auto", help="Transformers device_map. Use 'none' to disable.")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, help="Pass trust_remote_code=True.")
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--attn_implementation", default=None, help="Optional Transformers attention implementation.")

    # Tokenization/masking controls.
    parser.add_argument(
        "--add_special_tokens",
        action="store_true",
        help="Add tokenizer special tokens during forward. They are still excluded from counts unless --include_special_tokens is set.",
    )
    parser.add_argument(
        "--include_special_tokens",
        action="store_true",
        help="Count tokenizer special tokens in activation probabilities. By default they are excluded.",
    )

    # RUCAIBox methodology parameters.
    parser.add_argument("--positive_threshold", type=float, default=0.0, help="Count activation as active when activation > threshold.")
    parser.add_argument("--top_rate", type=float, default=0.01, help="RUCAIBox top_rate: fraction of lowest-entropy neurons to keep.")
    parser.add_argument(
        "--lape_percentile",
        type=float,
        default=None,
        help="Convenience alias for --top_rate. Example: 1.0 means top_rate=0.01.",
    )
    parser.add_argument(
        "--filter_rate",
        type=float,
        default=0.95,
        help="RUCAIBox filter_rate: global activation-probability kth-rate used before entropy top-k.",
    )
    parser.add_argument(
        "--activation_bar_ratio",
        type=float,
        default=0.95,
        help="RUCAIBox activation_bar_ratio: global activation-probability kth-rate for assigning selected neurons to dialects.",
    )
    parser.add_argument(
        "--activation_percentile",
        type=float,
        default=None,
        help=(
            "Convenience alias for both --filter_rate and --activation_bar_ratio. "
            "Example: 95.0 means 0.95. Use the explicit args if you want different values."
        ),
    )
    parser.add_argument(
        "--selection_threshold_scope",
        choices=["repo_global", "per_dialect"],
        default="repo_global",
        help=(
            "repo_global exactly matches identify.py: global activation_bar over all dialect/layer/neuron probs. "
            "per_dialect is optional diagnostic behavior, not the repo default."
        ),
    )

    # Hook controls.
    parser.add_argument(
        "--hook_projection",
        choices=["auto", "gate_proj", "up_proj", "dense_h_to_4h", "fc1", "c_fc"],
        default="auto",
        help="Projection to hook. auto chooses gate_proj when available, otherwise up_proj/dense_h_to_4h/fc1/c_fc.",
    )

    return parser.parse_args()


def dtype_from_arg(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def normalize_fieldnames(fieldnames: Optional[List[str]]) -> List[str]:
    if not fieldnames:
        return []
    result = []
    for name in fieldnames:
        cleaned = name.replace("\ufeff", "").strip()
        result.append(cleaned)
    return result


def read_manifest(path: str) -> List[ManifestEntry]:
    manifest_path = Path(path)
    base = manifest_path.parent
    entries: List[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "dialect" not in obj or "path" not in obj:
                raise ValueError(f"Manifest line {line_no} must contain 'dialect' and 'path': {line}")
            raw_path = Path(str(obj["path"]))
            if not raw_path.is_absolute():
                raw_path = base / raw_path
            entries.append(
                ManifestEntry(
                    dialect=str(obj["dialect"]),
                    path=str(raw_path),
                    lang=str(obj["lang"]) if obj.get("lang") is not None else None,
                )
            )
    if len(entries) < 2:
        raise ValueError("LAPE needs at least two dialects/languages in the manifest.")
    seen = set()
    for e in entries:
        if e.dialect in seen:
            raise ValueError(f"Duplicate dialect in manifest: {e.dialect}")
        seen.add(e.dialect)
    return entries


def sort_sentids(ids: Iterable[str], order: str, first_order: List[str], seed: int) -> List[str]:
    ids_set = set(ids)
    if order == "manifest":
        return [sid for sid in first_order if sid in ids_set]
    if order == "shuffle":
        out = list(ids_set)
        rng = random.Random(seed)
        rng.shuffle(out)
        return out
    if order == "lex":
        return sorted(ids_set)

    def numeric_key(x: str) -> Tuple[int, Any]:
        try:
            return (0, int(x))
        except Exception:
            try:
                return (0, float(x))
            except Exception:
                return (1, x)

    return sorted(ids_set, key=numeric_key)


def load_tsv_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], List[str], TsvStats]:
    path = Path(entry.path)
    if not path.exists():
        raise FileNotFoundError(f"TSV not found for dialect {entry.dialect}: {path}")
    split_re = re.compile(args.split_regex) if args.split_regex else None

    raw_rows = 0
    kept_rows = 0
    empty_text_rows = 0
    rows_by_id: Dict[str, str] = {}
    ordered_ids: List[str] = []
    duplicate_ids: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        reader.fieldnames = normalize_fieldnames(reader.fieldnames)
        if not reader.fieldnames:
            raise ValueError(f"Empty TSV or missing header: {path}")
        missing = [c for c in [args.id_col, args.text_col] if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {missing}; columns={reader.fieldnames}")
        if entry.lang is not None and not args.no_lang_filter and args.lang_col not in reader.fieldnames:
            raise ValueError(
                f"Manifest entry for {entry.dialect} specifies lang={entry.lang!r}, "
                f"but TSV lacks --lang_col {args.lang_col!r}. Use --no_lang_filter to disable."
            )
        if split_re is not None and args.split_col not in reader.fieldnames:
            raise ValueError(f"--split_regex was given, but TSV lacks split column {args.split_col!r}: {path}")

        for row in reader:
            raw_rows += 1
            # Normalize keys because DictReader keeps original keys in each row.
            row = {(k.replace("\ufeff", "").strip() if k is not None else k): v for k, v in row.items()}
            if entry.lang is not None and not args.no_lang_filter:
                if str(row.get(args.lang_col, "")) != entry.lang:
                    continue
            if split_re is not None:
                if not split_re.search(str(row.get(args.split_col, ""))):
                    continue
            sid = str(row.get(args.id_col, "")).strip()
            text = str(row.get(args.text_col, "")).strip()
            if not sid:
                continue
            if not text:
                empty_text_rows += 1
                continue
            kept_rows += 1
            if sid in rows_by_id:
                duplicate_ids.add(sid)
                if args.duplicate_policy == "error":
                    continue
                if args.duplicate_policy == "first":
                    continue
                if args.duplicate_policy == "last":
                    rows_by_id[sid] = text
                    continue
            else:
                rows_by_id[sid] = text
                ordered_ids.append(sid)

    if duplicate_ids and args.duplicate_policy == "error":
        sample = sorted(duplicate_ids)[:10]
        raise ValueError(
            f"Dialect {entry.dialect} has {len(duplicate_ids)} duplicate sentID values in {path}. "
            f"Sample={sample}. Use --duplicate_policy first or last to continue."
        )

    stats = TsvStats(
        dialect=entry.dialect,
        path=str(path),
        raw_rows=raw_rows,
        kept_rows=kept_rows,
        empty_text_rows=empty_text_rows,
        duplicate_sentids=len(duplicate_ids),
        unique_sentids=len(rows_by_id),
    )
    return rows_by_id, ordered_ids, stats


def load_parallel_data(args: argparse.Namespace) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, Any]]:
    entries = read_manifest(args.data_manifest)
    dialects = [e.dialect for e in entries]
    maps: Dict[str, Dict[str, str]] = {}
    first_order: List[str] = []
    tsv_stats: List[TsvStats] = []

    for idx, entry in enumerate(entries):
        sid_to_text, ordered_ids, stats = load_tsv_entry(entry, args)
        maps[entry.dialect] = sid_to_text
        if idx == 0:
            first_order = ordered_ids
        tsv_stats.append(stats)

    shared_ids = set(maps[dialects[0]].keys())
    for dialect in dialects[1:]:
        shared_ids &= set(maps[dialect].keys())

    if not shared_ids:
        raise ValueError("No shared sentIDs found across all dialect TSV files.")

    sorted_ids = sort_sentids(shared_ids, args.sentid_order, first_order, args.seed)
    if args.max_parallel_sentences is not None:
        sorted_ids = sorted_ids[: args.max_parallel_sentences]

    data_by_dialect = {
        dialect: [maps[dialect][sid] for sid in sorted_ids]
        for dialect in dialects
    }

    all_ids_union = set().union(*(set(m.keys()) for m in maps.values()))
    dropped_summary: Dict[str, Any] = {
        "union_sentids": len(all_ids_union),
        "shared_sentids_before_limit": len(shared_ids),
        "used_shared_sentids": len(sorted_ids),
        "missing_from_each_dialect": {
            dialect: len(all_ids_union - set(maps[dialect].keys())) for dialect in dialects
        },
        "not_used_after_intersection_or_limit": {
            dialect: len(set(maps[dialect].keys()) - set(sorted_ids)) for dialect in dialects
        },
    }

    data_stats = {
        "dialects": dialects,
        "manifest": [asdict(e) for e in entries],
        "tsv_stats": [asdict(s) for s in tsv_stats],
        "shared_sentids_before_limit": len(shared_ids),
        "used_parallel_sentids": len(sorted_ids),
        "dropped_summary": dropped_summary,
    }
    return dialects, sorted_ids, data_by_dialect, data_stats


def get_input_device(model: nn.Module) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        return next(emb.parameters()).device
    except Exception:
        return next(model.parameters()).device


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[nn.Module, Any, str]:
    model_id = resolve_model_id(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        # Causal LMs often have no pad token; using EOS as padding is standard for batched inference.
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad/eos/bos/unk token; cannot batch with padding.")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype_from_arg(args.dtype),
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer, model_id


def get_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    candidate_paths = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
        ("decoder", "layers"),
    ]
    for path in candidate_paths:
        obj: Any = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, (list, tuple, nn.ModuleList)) and len(obj) > 0:
            return obj
    raise ValueError(
        "Could not locate decoder layers. Expected one of: model.layers, transformer.h, "
        "gpt_neox.layers, model.decoder.layers."
    )


def get_mlp(layer: nn.Module) -> nn.Module:
    for name in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError(f"Could not find MLP module in layer type {type(layer)}")


def activation_from_name(name: Any) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    if name is None:
        return None
    key = str(name).lower()
    if key in ACT2FN:
        return ACT2FN[key]
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


def get_activation_fn(mlp: nn.Module, model_config: Any) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    for attr in ["act_fn", "activation_fn", "gelu_impl", "act", "activation"]:
        if hasattr(mlp, attr):
            fn = getattr(mlp, attr)
            if callable(fn):
                return fn
            named = activation_from_name(fn)
            if named is not None:
                return named
    for cfg_attr in ["hidden_act", "activation_function", "activation", "act_fn"]:
        act_name = getattr(model_config, cfg_attr, None)
        fn = activation_from_name(act_name)
        if fn is not None:
            return fn
    return None


def infer_intermediate_size(module: nn.Module) -> int:
    if hasattr(module, "out_features"):
        return int(getattr(module, "out_features"))
    # Fallback for some quantized/custom linears.
    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
        return int(weight.shape[0])
    raise ValueError(f"Could not infer intermediate size from projection module {module}")


def build_hook_specs(model: nn.Module, args: argparse.Namespace) -> List[HookSpec]:
    layers = get_decoder_layers(model)
    specs: List[HookSpec] = []
    for layer_id, layer in enumerate(layers):
        mlp = get_mlp(layer)
        projection_name: Optional[str] = None
        if args.hook_projection != "auto":
            if not hasattr(mlp, args.hook_projection):
                raise ValueError(f"Layer {layer_id} MLP lacks requested projection {args.hook_projection!r}")
            projection_name = args.hook_projection
        else:
            # Prefer gated MLP neuron definition, matching RUCAIBox LLaMA path.
            for name in ["gate_proj", "up_proj", "dense_h_to_4h", "fc1", "c_fc"]:
                if hasattr(mlp, name):
                    projection_name = name
                    break
        if projection_name is None:
            raise ValueError(f"Could not find a supported projection in layer {layer_id} MLP: {mlp}")

        projection = getattr(mlp, projection_name)
        activation_fn = get_activation_fn(mlp, model.config)
        if activation_fn is None:
            warnings.warn(
                f"Layer {layer_id}: no exposed activation function found. The hook will threshold raw "
                f"{projection_name} output; this may not equal post-activation neuron values.",
                RuntimeWarning,
            )
        intermediate_size = infer_intermediate_size(projection)
        specs.append(
            HookSpec(
                layer_id=layer_id,
                projection_name=projection_name,
                projection_module=projection,
                activation_fn=activation_fn,
                intermediate_size=intermediate_size,
            )
        )

    sizes = {s.intermediate_size for s in specs}
    if len(sizes) != 1:
        raise ValueError(f"Layers have non-uniform intermediate sizes: {sorted(sizes)}")
    return specs


def make_counting_hook(spec: HookSpec, state: ActivationState) -> Callable[..., None]:
    def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        if state.dialect_id is None or state.valid_mask is None:
            return
        acts = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(acts):
            raise TypeError(f"Projection hook output for layer {spec.layer_id} is not a tensor: {type(acts)}")
        if spec.activation_fn is not None:
            acts = spec.activation_fn(acts)
        if acts.dim() != 3:
            raise ValueError(f"Expected activation shape [batch, seq, inter], got {tuple(acts.shape)}")

        mask = state.valid_mask.to(device=acts.device, dtype=torch.bool)
        if mask.shape != acts.shape[:2]:
            raise ValueError(
                f"valid_mask shape {tuple(mask.shape)} does not match activation prefix {tuple(acts.shape[:2])}"
            )
        active = acts > state.positive_threshold
        counts = (active & mask.unsqueeze(-1)).sum(dim=(0, 1)).to(device="cpu", dtype=torch.int64)
        state.active_counts[spec.layer_id, :, state.dialect_id] += counts

    return hook


def register_counting_hooks(specs: Sequence[HookSpec], state: ActivationState) -> List[Any]:
    handles = []
    for spec in specs:
        handles.append(spec.projection_module.register_forward_hook(make_counting_hook(spec, state)))
    return handles


def special_tokens_mask_from_input_ids(input_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        mask |= input_ids.eq(token_id)
    return mask


def tokenize_batch(texts: List[str], tokenizer: Any, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.seq_len,
        add_special_tokens=args.add_special_tokens,
        return_special_tokens_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).bool()

    if args.include_special_tokens:
        valid_mask = attention_mask
    else:
        if "special_tokens_mask" in encoded:
            special_mask = encoded["special_tokens_mask"].bool()
        else:
            special_mask = special_tokens_mask_from_input_ids(input_ids, tokenizer)
        # Fallback catches special ids that return_special_tokens_mask may miss in some custom tokenizers.
        special_mask |= special_tokens_mask_from_input_ids(input_ids, tokenizer)
        valid_mask = attention_mask & (~special_mask)
    return input_ids, attention_mask, valid_mask


def collect_activation_counts(
    model: nn.Module,
    tokenizer: Any,
    dialects: List[str],
    data_by_dialect: Dict[str, List[str]],
    specs: Sequence[HookSpec],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_layers = len(specs)
    intermediate_size = specs[0].intermediate_size
    num_dialects = len(dialects)

    # Match repo's orientation: [num_layers, intermediate_size, num_dialects].
    active_counts = torch.zeros(num_layers, intermediate_size, num_dialects, dtype=torch.int64)
    token_counts = torch.zeros(num_dialects, dtype=torch.int64)
    state = ActivationState(active_counts=active_counts, positive_threshold=args.positive_threshold)
    handles = register_counting_hooks(specs, state)
    input_device = get_input_device(model)

    try:
        for dialect_id, dialect in enumerate(dialects):
            texts = data_by_dialect[dialect]
            pbar = tqdm(range(0, len(texts), args.batch_size), desc=f"Collecting {dialect}", dynamic_ncols=True)
            for start in pbar:
                batch_texts = texts[start : start + args.batch_size]
                input_ids, attention_mask, valid_mask = tokenize_batch(batch_texts, tokenizer, args)
                valid_tokens = int(valid_mask.sum().item())
                if valid_tokens == 0:
                    continue

                input_ids = input_ids.to(input_device)
                attention_mask = attention_mask.to(input_device)

                state.dialect_id = dialect_id
                state.valid_mask = valid_mask  # keep CPU copy; hook moves to activation device
                forward_ok = False
                try:
                    with torch.inference_mode():
                        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    forward_ok = True
                finally:
                    state.reset_batch()
                if forward_ok:
                    token_counts[dialect_id] += valid_tokens
    finally:
        for h in handles:
            h.remove()
    return active_counts, token_counts


def kth_percentile(values: torch.Tensor, ratio: float) -> torch.Tensor:
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"Percentile ratio must be in (0, 1], got {ratio}")
    flat = values.flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        raise ValueError("Cannot compute percentile of empty/non-finite tensor.")
    k = int(round(flat.numel() * ratio))
    k = max(1, min(k, flat.numel()))
    return flat.kthvalue(k).values


def rucai_identify(
    active_counts: torch.Tensor,
    token_counts: torch.Tensor,
    dialects: List[str],
    args: argparse.Namespace,
) -> Tuple[List[List[torch.LongTensor]], Dict[str, Any]]:
    """Replicate identify.py with optional per-dialect assignment threshold.

    Repo logic, in short:
      activation_probs = over_zero / n
      normed = activation_probs / activation_probs.sum(lang)
      entropy = -sum(normed * log(normed))
      set entropy to +inf if no language's activation prob exceeds global 95th percentile
      take bottom 1% entropy neurons
      assign selected neurons to language if selected prob exceeds global 95th percentile
    """
    if active_counts.dim() != 3:
        raise ValueError("active_counts must be [num_layers, intermediate_size, num_dialects]")
    num_layers, intermediate_size, num_dialects = active_counts.shape
    if num_dialects != len(dialects):
        raise ValueError("active_counts dialect dimension does not match dialect list")
    if torch.any(token_counts <= 0):
        bad = [dialects[i] for i, n in enumerate(token_counts.tolist()) if n <= 0]
        raise ValueError(f"Zero valid tokens for dialect(s): {bad}")

    n = token_counts.to(dtype=torch.float32)
    activation_probs = active_counts.to(dtype=torch.float32) / n.view(1, 1, num_dialects)

    normed = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed[torch.isnan(normed)] = 0
    log_probs = torch.where(normed > 0, normed.log(), torch.zeros_like(normed))
    entropy = -torch.sum(normed * log_probs, dim=-1)
    if torch.isnan(entropy).sum().item():
        raise ValueError("NaNs found in entropy")

    flattened_probs = activation_probs.flatten()
    top_prob_value = kth_percentile(flattened_probs, args.filter_rate).item()

    entropy_for_selection = entropy.clone()
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy_for_selection[top_position == 0] = torch.inf

    flattened_entropy = entropy_for_selection.flatten()
    top_entropy_count = int(round(flattened_entropy.numel() * args.top_rate))
    top_entropy_count = max(1, min(top_entropy_count, flattened_entropy.numel()))
    finite_count = int(torch.isfinite(flattened_entropy).sum().item())
    if finite_count == 0:
        warnings.warn(
            "No neuron passed the pre-filter `(activation_probs > top_prob_value).any(lang)`. "
            "The output neuron mask will be empty.",
            RuntimeWarning,
        )
        selected_flat_indices = torch.empty(0, dtype=torch.long)
    else:
        k = min(top_entropy_count, finite_count)
        _, selected_flat_indices = flattened_entropy.topk(k, largest=False)

    row_index = selected_flat_indices // intermediate_size
    col_index = selected_flat_indices % intermediate_size
    selected_probs = activation_probs[row_index, col_index] if selected_flat_indices.numel() else torch.empty(0, num_dialects)

    if args.selection_threshold_scope == "repo_global":
        activation_bar = kth_percentile(flattened_probs, args.activation_bar_ratio).item()
        selected_probs_by_dialect = selected_probs.transpose(0, 1) if selected_probs.numel() else torch.empty(num_dialects, 0)
        lang_ids, selected_indices_within_top = torch.where(selected_probs_by_dialect > activation_bar)
        activation_bars_by_dialect = {d: activation_bar for d in dialects}
    else:
        bars = []
        lang_chunks = []
        idx_chunks = []
        for d in range(num_dialects):
            bar_d = kth_percentile(activation_probs[:, :, d], args.activation_bar_ratio).item()
            bars.append(bar_d)
            if selected_probs.numel():
                idx = torch.where(selected_probs[:, d] > bar_d)[0]
                if idx.numel():
                    lang_chunks.append(torch.full_like(idx, d))
                    idx_chunks.append(idx)
        activation_bar = float("nan")
        activation_bars_by_dialect = {dialects[d]: float(bars[d]) for d in range(num_dialects)}
        if idx_chunks:
            lang_ids = torch.cat(lang_chunks)
            selected_indices_within_top = torch.cat(idx_chunks)
        else:
            lang_ids = torch.empty(0, dtype=torch.long)
            selected_indices_within_top = torch.empty(0, dtype=torch.long)

    merged_index = torch.stack((row_index, col_index), dim=-1) if selected_flat_indices.numel() else torch.empty(0, 2, dtype=torch.long)

    final_indices: List[List[torch.LongTensor]] = []
    selected_records: List[Dict[str, Any]] = []
    for dialect_id, dialect in enumerate(dialects):
        chosen = selected_indices_within_top[lang_ids == dialect_id]
        if chosen.numel():
            pairs = [tuple(x.tolist()) for x in merged_index[chosen]]
            pairs = sorted(set((int(l), int(h)) for l, h in pairs))
        else:
            pairs = []
        layer_lists: List[List[int]] = [[] for _ in range(num_layers)]
        for layer_id, neuron_id in pairs:
            layer_lists[layer_id].append(neuron_id)
            probs_all = activation_probs[layer_id, neuron_id].tolist()
            selected_records.append(
                {
                    "dialect": dialect,
                    "dialect_id": dialect_id,
                    "layer": layer_id,
                    "neuron": neuron_id,
                    "entropy": float(entropy[layer_id, neuron_id].item()),
                    "activation_probability": float(activation_probs[layer_id, neuron_id, dialect_id].item()),
                    "activation_probabilities_by_dialect": {
                        dialects[k]: float(probs_all[k]) for k in range(num_dialects)
                    },
                }
            )
        final_indices.append([torch.tensor(x, dtype=torch.long) for x in layer_lists])

    argmax_counts = [0 for _ in dialects]
    if selected_probs.numel():
        binc = torch.bincount(selected_probs.argmax(dim=-1), minlength=num_dialects).tolist()
        argmax_counts = [int(x) for x in binc]

    selected_counts = {dialects[i]: int(sum(len(t) for t in final_indices[i])) for i in range(num_dialects)}
    thresholds = {
        "top_rate": args.top_rate,
        "filter_rate": args.filter_rate,
        "activation_bar_ratio": args.activation_bar_ratio,
        "selection_threshold_scope": args.selection_threshold_scope,
        "top_prob_value_filter": float(top_prob_value),
        "activation_bar_global": float(activation_bar) if math.isfinite(activation_bar) else None,
        "activation_bars_by_dialect": activation_bars_by_dialect,
        "positive_threshold": args.positive_threshold,
        "top_entropy_count_requested": int(top_entropy_count),
        "top_entropy_count_used": int(selected_flat_indices.numel()),
        "prefilter_finite_neurons": int(finite_count),
        "selected_low_entropy_argmax_counts": {dialects[i]: argmax_counts[i] for i in range(num_dialects)},
        "selected_counts": selected_counts,
    }
    details = {
        "activation_probs": activation_probs,
        "entropy": entropy,
        "entropy_for_selection": entropy_for_selection,
        "selected_flat_indices": selected_flat_indices,
        "selected_records": selected_records,
        "thresholds": thresholds,
    }
    return final_indices, details


def tensor_distribution(x: torch.Tensor, name: str) -> Dict[str, Optional[float]]:
    y = x.detach().flatten().to(dtype=torch.float32)
    y = y[torch.isfinite(y)]
    if y.numel() == 0:
        return {"name": name, "count": 0}
    qs = torch.tensor([0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0], dtype=torch.float32)
    vals = torch.quantile(y, qs).tolist()
    return {
        "name": name,
        "count": int(y.numel()),
        "min": float(vals[0]),
        "p01": float(vals[1]),
        "p05": float(vals[2]),
        "p10": float(vals[3]),
        "median": float(vals[4]),
        "p90": float(vals[5]),
        "p95": float(vals[6]),
        "p99": float(vals[7]),
        "max": float(vals[8]),
        "mean": float(y.mean().item()),
    }


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_selected_csv(path: Path, selected_records: List[Dict[str, Any]], dialects: List[str]) -> None:
    fieldnames = ["dialect", "dialect_id", "layer", "neuron", "entropy", "activation_probability"] + [
        f"p_{d}" for d in dialects
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in selected_records:
            row = {k: rec.get(k) for k in ["dialect", "dialect_id", "layer", "neuron", "entropy", "activation_probability"]}
            for d in dialects:
                row[f"p_{d}"] = rec["activation_probabilities_by_dialect"][d]
            writer.writerow(row)


def save_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    model_id: str,
    dialects: List[str],
    sentids: List[str],
    data_stats: Dict[str, Any],
    specs: Sequence[HookSpec],
    active_counts: torch.Tensor,
    token_counts: torch.Tensor,
    final_indices: List[List[torch.LongTensor]],
    details: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    activation_probs = details["activation_probs"]
    entropy = details["entropy"]
    entropy_for_selection = details["entropy_for_selection"]
    selected_records = details["selected_records"]
    thresholds = details["thresholds"]

    # RUCAIBox-compatible main artifact.
    torch.save(final_indices, out_dir / "neurons.pth")

    # Raw stats and scores.
    torch.save({"n": token_counts, "over_zero": active_counts}, out_dir / "activation_counts_combined.pt")
    torch.save(active_counts, out_dir / "over_zero.pt")
    torch.save(token_counts, out_dir / "token_counts.pt")
    torch.save(activation_probs, out_dir / "activation_probs.pt")
    torch.save(entropy, out_dir / "entropy.pt")
    torch.save(entropy_for_selection, out_dir / "entropy_for_selection.pt")

    for d_id, dialect in enumerate(dialects):
        torch.save(
            {"n": int(token_counts[d_id].item()), "over_zero": active_counts[:, :, d_id].clone()},
            out_dir / f"activation.{dialect}.pt",
        )

    save_json(out_dir / "selected_neurons.json", selected_records)
    save_selected_csv(out_dir / "selected_neurons.csv", selected_records, dialects)
    save_json(out_dir / "thresholds.json", thresholds)
    save_json(out_dir / "dialects.json", {str(i): d for i, d in enumerate(dialects)})
    save_json(out_dir / "token_counts.json", {dialects[i]: int(token_counts[i].item()) for i in range(len(dialects))})
    save_json(out_dir / "data_stats.json", data_stats)
    save_json(out_dir / "dropped_sentids_summary.json", data_stats.get("dropped_summary", {}))

    with (out_dir / "parallel_sentids_used.txt").open("w", encoding="utf-8") as f:
        for sid in sentids:
            f.write(str(sid) + "\n")

    hook_info = [
        {
            "layer": s.layer_id,
            "projection_name": s.projection_name,
            "intermediate_size": s.intermediate_size,
            "activation_fn": getattr(s.activation_fn, "__name__", str(s.activation_fn)) if s.activation_fn is not None else None,
        }
        for s in specs
    ]
    save_json(out_dir / "model_hook_info.json", hook_info)

    diagnostics = {
        "entropy_distribution": tensor_distribution(entropy, "entropy"),
        "entropy_for_selection_distribution": tensor_distribution(entropy_for_selection, "entropy_for_selection"),
        "activation_probability_distribution_global": tensor_distribution(activation_probs, "activation_probs"),
        "activation_probability_distribution_by_dialect": {
            dialects[i]: tensor_distribution(activation_probs[:, :, i], f"activation_probs_{dialects[i]}")
            for i in range(len(dialects))
        },
        "max_entropy": math.log(len(dialects)),
    }
    save_json(out_dir / "score_diagnostics.json", diagnostics)

    summary = {
        "model_id": model_id,
        "num_dialects": len(dialects),
        "dialects": dialects,
        "num_layers": active_counts.shape[0],
        "intermediate_size": active_counts.shape[1],
        "parallel_sentences_used": len(sentids),
        "token_counts": {dialects[i]: int(token_counts[i].item()) for i in range(len(dialects))},
        "thresholds": thresholds,
        "selected_counts": thresholds["selected_counts"],
        "main_artifact": "neurons.pth",
    }
    save_json(out_dir / "run_summary.json", summary)

    config = vars(args).copy()
    config["resolved_model_id"] = model_id
    save_json(out_dir / "config_used.json", config)


def main() -> None:
    args = parse_args()
    if args.lape_percentile is not None:
        args.top_rate = args.lape_percentile / 100.0
    if args.activation_percentile is not None:
        rate = args.activation_percentile / 100.0
        args.filter_rate = rate
        args.activation_bar_ratio = rate

    if args.batch_size <= 0 or args.seq_len <= 0:
        raise ValueError("--batch_size and --seq_len must be positive.")
    for name in ["top_rate", "filter_rate", "activation_bar_ratio"]:
        val = getattr(args, name)
        if not (0.0 < val <= 1.0):
            raise ValueError(f"--{name} must be in (0, 1], got {val}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading parallel TSV data...")
    dialects, sentids, data_by_dialect, data_stats = load_parallel_data(args)
    log(f"Dialects: {dialects}")
    log(f"Parallel sentIDs used: {len(sentids)}")

    log("Loading model and tokenizer...")
    model, tokenizer, model_id = load_model_and_tokenizer(args)
    log(f"Resolved model ID: {model_id}")

    specs = build_hook_specs(model, args)
    log(
        f"Hooking {len(specs)} layers; projection={specs[0].projection_name}; "
        f"intermediate_size={specs[0].intermediate_size}"
    )

    log("Collecting activation counts...")
    active_counts, token_counts = collect_activation_counts(model, tokenizer, dialects, data_by_dialect, specs, args)

    log("Identifying neurons with RUCAIBox LAPE selection logic...")
    final_indices, details = rucai_identify(active_counts, token_counts, dialects, args)

    log("Saving outputs...")
    save_outputs(
        out_dir=out_dir,
        args=args,
        model_id=model_id,
        dialects=dialects,
        sentids=sentids,
        data_stats=data_stats,
        specs=specs,
        active_counts=active_counts,
        token_counts=token_counts,
        final_indices=final_indices,
        details=details,
    )
    log(f"Done. Main artifact: {out_dir / 'neurons.pth'}")
    log(f"Selected counts: {details['thresholds']['selected_counts']}")


if __name__ == "__main__":
    main()