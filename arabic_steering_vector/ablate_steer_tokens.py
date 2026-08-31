"""
Ablate the number of response tokens that receive activation steering.

For a fixed steering vector / coefficient / layer, sweep over a list of N values
and measure how dialect authenticity and coherence change as a function of N.

During each run:
  - Prefill behavior is unchanged (follows --steering-type, same as always).
  - Decoding: only the first N generated tokens are steered; tokens N+1 onwards
    are produced with the unmodified model.

Usage:
    python ablate_steer_tokens.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --vector-path dialect_vectors/Qwen2.5-7B-Instruct/Egyptian_response_avg_diff.pt \
        --steer-dialect Egyptian \
        --layer 16 --coef 3.0 \
        --prompt "كيف حالك اليوم؟" \
        --n-steer-tokens 0 1 2 5 10 20 50 all
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steer import ActivationSteerer
from steer_dialect_and_compare import (
    MSA_EVAL_PROMPT,
    COHERENCE_PROMPT,
    ARABIC_FLUENCY_PROMPT,
    make_city_eval_prompt,
    judge_with_model,
)


def generate_response(model, tokenizer, prompt, vector, layer, coef,
                      positions, n_steer_tokens, max_new_tokens, temperature):
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id,
    )

    if n_steer_tokens == 0:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
    else:
        steerer_kwargs = dict(
            coeff=coef,
            layer_idx=layer - 1,
            positions=positions,
            n_steer_tokens=n_steer_tokens,  # None means steer all decoding steps
        )
        with ActivationSteerer(model, vector, **steerer_kwargs):
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate_response(model, tokenizer, question, answer, eval_metrics):
    return {
        name: judge_with_model(model, tokenizer, prompt, question, answer)
        for name, prompt in eval_metrics.items()
    }


def parse_n_steer_tokens(raw_values):
    """Convert CLI strings to int or None ('all' -> None, '0' -> 0, etc.)."""
    result = []
    for v in raw_values:
        if v.lower() == "all":
            result.append(None)
        else:
            n = int(v)
            if n < 0:
                raise argparse.ArgumentTypeError(f"n_steer_tokens must be >= 0, got {n}")
            result.append(n)
    return result


def build_eval_metrics(metric_names):
    metrics = {}
    for name in metric_names:
        if name == "coherence":
            metrics["coherence"] = COHERENCE_PROMPT
        elif name == "arabic_fluency":
            metrics["arabic_fluency"] = ARABIC_FLUENCY_PROMPT
        elif name == "MSA":
            metrics["MSA"] = MSA_EVAL_PROMPT
        else:
            metrics[name] = make_city_eval_prompt(name)
    return metrics


def print_results_table(results, metrics):
    col_w = 12
    print(f"\n{'='*80}")
    print("ABLATION RESULTS")
    print(f"{'='*80}")
    header = f"{'n_steer':>10}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(header)
    print("─" * len(header))
    for r in results:
        row = f"{r['label']:>10}"
        for m in metrics:
            s = r["scores"].get(m)
            row += f"{(f'{s:.1f}' if s is not None else 'N/A'):>{col_w}}"
        print(row)
    print("─" * len(header))


def print_responses(results):
    print(f"\n{'='*80}")
    print("GENERATED RESPONSES")
    print(f"{'='*80}")
    for r in results:
        print(f"\n--- n_steer_tokens={r['label']} ---")
        print(r["response"])


def main():
    parser = argparse.ArgumentParser(
        description="Ablate number of steered response tokens and measure dialect scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ablate_steer_tokens.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --vector-path dialect_vectors/Qwen2.5-7B-Instruct/Egyptian_response_avg_diff.pt \\
      --steer-dialect Egyptian \\
      --layer 16 --coef 3.0 \\
      --prompt "كيف حالك اليوم؟" \\
      --n-steer-tokens 0 1 2 5 10 20 50 all
        """,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--vector-path", required=True, help="Path to dialect vector .pt file")
    parser.add_argument("--steer-dialect", required=True, help="Name of the target dialect")
    parser.add_argument("--layer", type=int, required=True, help="Layer to apply steering")
    parser.add_argument("--coef", type=float, default=3.0, help="Steering coefficient (default: 3.0)")
    parser.add_argument("--prompt", type=str, default=None, help="Arabic prompt (interactive if omitted)")
    parser.add_argument(
        "--n-steer-tokens",
        nargs="+",
        default=["0", "1", "2", "5", "10", "20", "50", "all"],
        metavar="N",
        help="Number of response tokens to steer. 'all' means no limit. (default: 0 1 2 5 10 20 50 all)",
    )
    parser.add_argument(
        "--eval-dialects",
        nargs="+",
        default=None,
        help="Metrics to evaluate. Options: dialect names, MSA, coherence, arabic_fluency. "
             "Defaults to [--steer-dialect, coherence, arabic_fluency].",
    )
    parser.add_argument(
        "--steering-type",
        choices=["all", "prompt", "response"],
        default="response",
        help="Positions mode during prefill / decoding (default: response)",
    )
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")

    args = parser.parse_args()

    n_values = parse_n_steer_tokens(args.n_steer_tokens)

    if args.eval_dialects is None:
        args.eval_dialects = [args.steer_dialect, "coherence", "arabic_fluency"]

    print(f"\n{'='*80}")
    print("RESPONSE TOKEN STEERING ABLATION")
    print(f"{'='*80}")
    print(f"Model:           {args.model}")
    print(f"Target dialect:  {args.steer_dialect}")
    print(f"Layer:           {args.layer}")
    print(f"Coefficient:     {args.coef}")
    print(f"Steering type:   {args.steering_type}")
    print(f"N values:        {[str(v) if v is not None else 'all' for v in n_values]}")
    print(f"Eval metrics:    {', '.join(args.eval_dialects)}")
    print(f"{'='*80}\n")

    print("Loading model...")
    load_kwargs = {"torch_dtype": torch.float16}
    try:
        import accelerate  # noqa: F401
        load_kwargs["device_map"] = "auto"
    except ImportError:
        print("  Note: accelerate not found, loading without device_map")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on: {next(model.parameters()).device}")

    print(f"\nLoading vector: {args.vector_path}")
    all_layers = torch.load(args.vector_path, weights_only=False)
    vector = all_layers[args.layer]
    print(f"Vector shape: {vector.shape}")

    eval_metrics = build_eval_metrics(args.eval_dialects)

    if args.prompt:
        prompt = args.prompt
    else:
        print(f"\n{'─'*80}")
        prompt = input("Enter your Arabic prompt: ").strip()
        if not prompt:
            print("Empty prompt. Exiting.")
            return
    print(f"\nPrompt: {prompt}")

    results = []
    for n in n_values:
        label = "all" if n is None else str(n)
        print(f"\n{'='*80}")
        print(f"Generating: n_steer_tokens={label}")
        print(f"{'='*80}")

        response = generate_response(
            model, tokenizer, prompt, vector,
            layer=args.layer,
            coef=args.coef,
            positions=args.steering_type,
            n_steer_tokens=n,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"Response:\n{response}")

        print("\nEvaluating...")
        scores = evaluate_response(model, tokenizer, prompt, response, eval_metrics)
        for metric, score in scores.items():
            print(f"  {metric:<25} {f'{score:.1f}' if score is not None else 'N/A'}")

        results.append({"n": n, "label": label, "response": response, "scores": scores})

    print_results_table(results, list(eval_metrics.keys()))
    print_responses(results)


if __name__ == "__main__":
    main()
