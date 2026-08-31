"""
Generate baseline and steered responses for eval data, sweeping over layers.

Reads prompts from an eval JSONL file (eval_data/*.jsonl), generates a baseline
response and a steered response for each sample, and writes results to a JSONL
file per layer under results_layers/.

Output file naming: results_layers/{dialect}_layer{N}.jsonl

Each output line:
    {
        "sample_input":       <prompt from eval file>,
        "source":             <dialect code>,
        "layer":              <layer index>,
        "baseline_response":  <unsteered generation>,
        "steered_response":   <steered generation>
    }

Usage:
    # Single eval file, specific layers:
    python generate_steered_responses.py \\
        --model QCRI/Fanar-1-9B-Instruct \\
        --eval-file eval_data/egy.jsonl \\
        --vector-path dialect_vectors/Fanar-1-9B-Instruct/Cairo_response_avg_diff.pt \\
        --layers 8 12 16 20 24 \\
        --coef 3.0

    # All layers:
    python generate_steered_responses.py \\
        --model QCRI/Fanar-1-9B-Instruct \\
        --eval-file eval_data/egy.jsonl \\
        --vector-path dialect_vectors/Fanar-1-9B-Instruct/Cairo_response_avg_diff.pt \\
        --all-layers \\
        --coef 3.0
"""

import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steer import ActivationSteerer

EXTRACTION_PROMPT = "أجب بجملة واحدة."


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_eval_samples(eval_file):
    samples = []
    with open(eval_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ─── Generation ───────────────────────────────────────────────────────────────

def format_prompt(tokenizer, sample_prompt):
    messages = [{"role": "user", "content": f"{EXTRACTION_PROMPT}\n{sample_prompt}"}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{EXTRACTION_PROMPT}\n{sample_prompt}\n"


def generate(model, tokenizer, prompt_text, max_new_tokens, temperature):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_steered(model, tokenizer, prompt_text, vector, layer_idx, coef,
                     positions, max_new_tokens, temperature):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer_idx, positions=positions):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline and steered responses across layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--eval-file", required=True, help="Path to eval JSONL file (eval_data/*.jsonl)")
    parser.add_argument("--vector-path", required=True, help="Path to dialect vector .pt file")
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Layer indices to run (e.g. --layers 8 12 16 20). Ignored if --all-layers is set.",
    )
    parser.add_argument(
        "--all-layers", action="store_true",
        help="Run on every layer of the model (overrides --layers)",
    )
    parser.add_argument("--coef", type=float, default=3.0, help="Steering coefficient (default: 3.0)")
    parser.add_argument(
        "--steering-type", choices=["all", "prompt", "response"], default="response",
        help="Positions mode (default: response)",
    )
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy)")
    parser.add_argument("--output-dir", default="results_layers", help="Output directory (default: results_layers)")

    args = parser.parse_args()

    # ─── Load eval data ───────────────────────────────────────────────────────
    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"Error: eval file not found: {eval_file}")
        return

    samples = load_eval_samples(eval_file)
    if not samples:
        print("Error: no samples loaded.")
        return

    dialect = samples[0].get("dialect", eval_file.stem)
    print(f"\nDialect:  {dialect}")
    print(f"Samples:  {len(samples)}")

    # ─── Load model ───────────────────────────────────────────────────────────
    print("\nLoading model...")
    load_kwargs = {"torch_dtype": torch.float16}
    try:
        import accelerate  # noqa: F401
        load_kwargs["device_map"] = "auto"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on: {next(model.parameters()).device}")

    # ─── Load vectors ─────────────────────────────────────────────────────────
    print(f"\nLoading vectors: {args.vector_path}")
    all_layer_vectors = torch.load(args.vector_path, weights_only=False)
    num_layers = all_layer_vectors.shape[0]
    print(f"Vector shape: {all_layer_vectors.shape}  ({num_layers} layers)")

    # ─── Determine layers to run ──────────────────────────────────────────────
    if args.all_layers:
        layers = list(range(num_layers))
    elif args.layers:
        layers = args.layers
    else:
        print("Error: specify --layers or --all-layers")
        return

    print(f"Layers to process: {layers}")

    # ─── Output dir ───────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Generate baselines (once per sample) ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating baseline responses...")
    print(f"{'='*60}")

    formatted_prompts = [format_prompt(tokenizer, s["prompt"]) for s in samples]
    baselines = []
    for i, (sample, prompt_text) in enumerate(tqdm(zip(samples, formatted_prompts), total=len(samples))):
        response = generate(model, tokenizer, prompt_text, args.max_tokens, args.temperature)
        baselines.append(response)

    # ─── For each layer: generate steered responses and write JSONL ───────────
    for layer_idx in layers:
        if layer_idx >= num_layers:
            print(f"Skipping layer {layer_idx} — out of range (model has {num_layers} layers)")
            continue

        vector = all_layer_vectors[layer_idx]

        print(f"\n{'='*60}")
        print(f"Layer {layer_idx} — generating steered responses...")
        print(f"{'='*60}")

        out_path = Path(args.output_dir) / f"{dialect}_layer{layer_idx}.jsonl"
        with open(out_path, "w", encoding="utf-8") as out_f:
            for sample, prompt_text, baseline in tqdm(
                zip(samples, formatted_prompts, baselines), total=len(samples)
            ):
                steered = generate_steered(
                    model, tokenizer, prompt_text,
                    vector=vector,
                    layer_idx=layer_idx - 1,  # 0-indexed hook
                    coef=args.coef,
                    positions=args.steering_type,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                record = {
                    "sample_input": sample["prompt"],
                    "source": dialect,
                    "layer": layer_idx,
                    "baseline_response": baseline,
                    "steered_response": steered,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  Saved: {out_path}  ({len(samples)} records)")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
