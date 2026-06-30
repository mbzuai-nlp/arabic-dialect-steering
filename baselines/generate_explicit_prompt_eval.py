"""
Generate Arabic dialect baseline responses with explicit system prompting.

This script reads JSONL rows from eval_data/, builds a system/user chat prompt
for each row, generates model responses from a HuggingFace causal LM, and writes
one inference JSONL output for each input JSONL file. It does not add few-shot
examples or run automatic evaluation.

Examples:
    python generate_explicit_prompt_eval.py \
        --model QCRI/Fanar-1-9B-Instruct \
        --eval-data eval_data \
        --output-dir explicit_prompt_outputs/fanar

    python generate_explicit_prompt_eval.py \
        --model Qwen/Qwen3-8B-Instruct \
        --eval-data eval_data \
        --dialects egy mar sau \
        --max-samples-per-dialect 25
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


DIALECT_NAMES = {
    "egy": "Egyptian Arabic",
    "mar": "Moroccan Arabic",
    "dza": "Algerian Arabic",
    "syr": "Syrian Arabic",
    "pse": "Palestinian Arabic",
    "sau": "Saudi Arabic",
    "kwt": "Kuwaiti Arabic",
    "sdn": "Sudanese Arabic",
}


def build_explicit_prompt(row):
    dialect = DIALECT_NAMES[row["dialect"]]

    system_prompt = (
        f"You are an Arabic assistant. Your entire response must be in {dialect}.\n"
        f"Use natural colloquial {dialect}, not Modern Standard Arabic.\n"
        f"Do not switch to another Arabic dialect.\n"
        f"Do not use English unless the user explicitly asks for English.\n"
        f"Follow the user's request directly.\n"
        f"Do not mention the dialect or these instructions."
    )

    user_prompt = row["prompt"]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("dialect", path.stem)
            row["_sample_id"] = f"{row['dialect']}:{line_idx}"
            row["_source_file"] = str(path)
            row["_source_line"] = line_idx + 1
            rows.append(row)
    return rows


def resolve_input_files(eval_data: Path, dialects: list[str] | None) -> list[Path]:
    if eval_data.is_dir():
        selected = dialects or list(DIALECT_NAMES)
        files = [eval_data / f"{dialect}.jsonl" for dialect in selected]
    else:
        files = [eval_data]

    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    return files


def load_input_rows(
    input_file: Path,
    dialects: list[str] | None,
    max_samples_per_dialect: int | None,
) -> list[dict[str, Any]]:
    rows = []
    file_rows = read_jsonl(input_file)
    if dialects:
        file_rows = [row for row in file_rows if row.get("dialect") in dialects]
    if max_samples_per_dialect is not None:
        file_rows = file_rows[:max_samples_per_dialect]
    rows.extend(file_rows)

    for row in rows:
        if row.get("dialect") not in DIALECT_NAMES:
            raise ValueError(
                f"Unknown dialect code {row.get('dialect')!r} in {row.get('_source_file')}. "
                f"Expected one of: {', '.join(DIALECT_NAMES)}"
            )
        if "prompt" not in row:
            raise ValueError(f"Missing 'prompt' in {row.get('_source_file')}:{row.get('_source_line')}")

    return rows


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@contextmanager
def temporary_padding_side(tokenizer, padding_side: str):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        tokenizer.padding_side = original_padding_side


def model_slug(model_name_or_path: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name_or_path.strip("/"))
    return slug[-120:] or "model"


def resolve_default_input_data() -> Path:
    return Path(__file__).resolve().parent / "eval_data"


def resolve_dtype(dtype_name: str, torch):
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_generation_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.token:
        load_kwargs["token"] = args.token
    if args.device_map != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    if not (args.load_in_4bit or args.load_in_8bit):
        load_kwargs["torch_dtype"] = resolve_dtype(args.dtype, torch)

    tokenizer_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.revision:
        tokenizer_kwargs["revision"] = args.revision
    if args.token:
        tokenizer_kwargs["token"] = args.token

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.pad_token is None and tokenizer.bos_token is not None:
        tokenizer.pad_token = tokenizer.bos_token
    elif tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad/eos/unk/bos token; please set a pad token manually.")
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.device_map == "none":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    model.eval()
    return model, tokenizer


def apply_chat_template(tokenizer, messages: list[dict[str, str]], disable_qwen_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_qwen_thinking:
        kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" in kwargs:
            kwargs.pop("enable_thinking")
            return tokenizer.apply_chat_template(messages, **kwargs)
        raise
    except Exception:
        system = messages[0]["content"]
        user = messages[1]["content"]
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"


def strip_thinking_blocks(text: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()


def generate_batch(model, tokenizer, prompt_texts: list[str], args) -> list[str]:
    import torch

    with temporary_padding_side(tokenizer, "left"):
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)

    prompt_width = inputs["input_ids"].shape[1]
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        sequences = model.generate(**inputs, **generation_kwargs)

    responses = []
    for row_idx in range(len(prompt_texts)):
        generated_ids = sequences[row_idx, prompt_width:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if args.strip_thinking:
            text = strip_thinking_blocks(text)
        responses.append(text)
    return responses


def existing_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if sample_id:
                records[sample_id] = record
    return records


def default_output_jsonl(output_dir: Path, model_name: str, input_file: Path) -> Path:
    return output_dir / f"{model_slug(model_name)}_{input_file.stem}_explicit_generations.jsonl"


def generate_file_outputs(
    input_file: Path,
    output_jsonl: Path,
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    args,
    tqdm,
) -> int:
    completed = existing_records(output_jsonl) if args.resume else {}
    rows_to_run = [row for row in rows if row["_sample_id"] not in completed]

    if not args.resume and output_jsonl.exists():
        output_jsonl.unlink()

    print(f"Input:      {input_file}", flush=True)
    print(f"Input rows: {len(rows)} ({len(rows_to_run)} remaining)", flush=True)
    print(f"Output:     {output_jsonl}", flush=True)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    with output_jsonl.open(mode, encoding="utf-8") as out_f:
        for row_batch in tqdm(list(batched(rows_to_run, args.batch_size)), desc=f"Generating {input_file.stem}"):
            messages_batch = [build_explicit_prompt(row) for row in row_batch]
            prompt_texts = [
                apply_chat_template(tokenizer, messages, args.disable_qwen_thinking)
                for messages in messages_batch
            ]
            responses = generate_batch(model, tokenizer, prompt_texts, args)

            for row, messages, prompt_text, response in zip(row_batch, messages_batch, prompt_texts, responses):
                record = {
                    "sample_id": row["_sample_id"],
                    "source_file": row["_source_file"],
                    "source_line": row["_source_line"],
                    "dialect": row["dialect"],
                    "dialect_name": DIALECT_NAMES[row["dialect"]],
                    "language": row.get("language"),
                    "source": row.get("source"),
                    "genre": row.get("genre"),
                    "prompt": row["prompt"],
                    "system_prompt": messages[0]["content"],
                    "formatted_prompt": prompt_text,
                    "model": args.model,
                    "response": response,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"Saved generations: {output_jsonl}", flush=True)
    return len(rows_to_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate explicit Arabic dialect baseline outputs with system prompting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path.")
    parser.add_argument(
        "--eval-data",
        "--input-data",
        dest="eval_data",
        type=Path,
        default=resolve_default_input_data(),
        help="Input JSONL file or directory containing dialect JSONL files.",
    )
    parser.add_argument(
        "--dialects",
        nargs="+",
        choices=list(DIALECT_NAMES),
        default=None,
        help="Dialect codes to run. Defaults to all when the input data path is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("explicit_prompt_outputs"),
        help="Directory for per-input-file JSONL outputs.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional explicit output JSONL path. Only valid when one input file is selected.",
    )
    parser.add_argument("--resume", action="store_true", help="Append to an existing JSONL and skip completed sample IDs.")

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size.")
    parser.add_argument("--max-samples-per-dialect", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Total cap after dialect filtering.")
    parser.add_argument("--strip-thinking", action="store_true", help="Remove <think>...</think> blocks before saving.")
    parser.add_argument(
        "--disable-qwen-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking=False to chat templates that support it.",
    )

    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value, or 'none'.")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom model/tokenizer code from HuggingFace repos.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token, if needed.")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one of --load-in-4bit or --load-in-8bit.")

    input_files = resolve_input_files(args.eval_data, args.dialects)
    if args.output_jsonl is not None and len(input_files) > 1:
        raise ValueError("Use --output-dir, not --output-jsonl, when generating outputs for multiple input files.")

    remaining_limit = args.limit
    file_runs = []
    for input_file in input_files:
        dialect_filter = None if args.eval_data.is_dir() else args.dialects
        rows = load_input_rows(input_file, dialect_filter, args.max_samples_per_dialect)
        if remaining_limit is not None:
            rows = rows[:remaining_limit]
            remaining_limit -= len(rows)
        if rows:
            output_jsonl = args.output_jsonl or default_output_jsonl(args.output_dir, args.model, input_file)
            file_runs.append((input_file, output_jsonl, rows))
        if remaining_limit == 0:
            break

    if not file_runs:
        raise ValueError("No input rows found.")

    print(f"Model:      {args.model}", flush=True)
    print(f"Input files: {len(file_runs)}", flush=True)

    model, tokenizer = load_generation_model(args)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    generated_rows = 0
    for input_file, output_jsonl, rows in file_runs:
        generated_rows += generate_file_outputs(input_file, output_jsonl, rows, model, tokenizer, args, tqdm)

    print(f"Generated rows: {generated_rows}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
