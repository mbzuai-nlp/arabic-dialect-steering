"""
Generate steered responses for eval data, sweeping over layers 12-21
and coefficient values 1-5.

Reads prompts from an eval JSONL file (eval_data/*.jsonl), generates a steered
response for each sample, and writes results to a JSONL file per (layer, coeff)
pair under results_coeff/{model_name}/.

Output file naming: results_coeff/{model_name}/{dialect}_layer{N}_coeff{C}.jsonl

Each output line:
    {
        "sample_input":    <prompt from eval file>,
        "dialect":         <dialect code, e.g. "egy">,
        "source":          <dataset source, e.g. "MADAR-26">,
        "layer":           <layer index>,
        "coeff":           <steering coefficient>,
        "steered_response": <steered generation>
    }

Usage:
    python generate_steered_responses_coeff_sweep.py \\
        --model QCRI/Fanar-1-9B-Instruct \\
        --eval-file eval_data/egy.jsonl \\
        --vector-path dialect_vectors/Fanar-1-9B-Instruct/Cairo_response_avg_diff.pt
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

SWEEP_LAYERS = list(range(24, 25))   # 12 to 21 inclusive
SWEEP_COEFFS = [4]


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
        description="Generate baseline and steered responses sweeping layers 12-21 and coefficients 1-5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--eval-file", required=True, help="Path to eval JSONL file (eval_data/*.jsonl)")
    parser.add_argument("--vector-path", required=True, help="Path to dialect vector .pt file")
    parser.add_argument(
        "--steering-type", choices=["all", "prompt", "response"], default="response",
        help="Positions mode (default: response)",
    )
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy)")
    parser.add_argument("--output-dir", default="results_coeff", help="Output root directory (default: results_coeff)")

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

    # ─── Output dir: results_coeff/{model_name}/ ──────────────────────────────
    model_name = Path(args.model).name
    out_dir = Path(args.output_dir) / model_name
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    print(f"Layers:      {SWEEP_LAYERS}")
    print(f"Coefficients: {SWEEP_COEFFS}")

    formatted_prompts = [format_prompt(tokenizer, s["prompt"]) for s in samples]

    # ─── Sweep layers × coefficients ──────────────────────────────────────────
    for layer_idx in SWEEP_LAYERS:
        if layer_idx >= num_layers:
            print(f"Skipping layer {layer_idx} — out of range (model has {num_layers} layers)")
            continue

        vector = all_layer_vectors[layer_idx]

        for coef in SWEEP_COEFFS:
            print(f"\n{'='*60}")
            print(f"Layer {layer_idx}  |  Coeff {coef} — generating steered responses...")
            print(f"{'='*60}")

            out_path = out_dir / f"{dialect}_layer{layer_idx}_coeff{coef}.jsonl"
            with open(out_path, "w", encoding="utf-8") as out_f:
                for sample, prompt_text in tqdm(
                    zip(samples, formatted_prompts), total=len(samples)
                ):
                    steered = generate_steered(
                        model, tokenizer, prompt_text,
                        vector=vector,
                        layer_idx=layer_idx - 1,  # 0-indexed hook
                        coef=coef,
                        positions=args.steering_type,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    record = {
                        "sample_input": sample["prompt"],
                        "dialect": sample.get("dialect", dialect),
                        "source": sample.get("source", ""),
                        "layer": layer_idx,
                        "coeff": coef,
                        "steered_response": steered,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"  Saved: {out_path}  ({len(samples)} records)")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
