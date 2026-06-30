"""
Steer a model towards an Arabic dialect and compare dialect authenticity before/after.

Pipeline:
  1. Load model + dialect steering vector
  2. User provides an Arabic prompt
  3. Generate baseline response (no steering)
  4. Evaluate baseline: dialect authenticity + coherence
  5. Apply steering towards target dialect
  6. Generate steered response (same prompt)
  7. Evaluate steered response on same metrics
  8. Print comparison

Usage:
    python steer_dialect_and_compare.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --vector-path dialect_vectors/Qwen2.5-7B-Instruct/Levantine_response_avg_diff.pt \\
        --steer-dialect Levantine \\
        --layer 16 --coef 3.0 \\
        --prompt "كيف حالك اليوم؟"

    # Interactive mode (prompts for input):
    python steer_dialect_and_compare.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --vector-path dialect_vectors/Qwen2.5-7B-Instruct/Egyptian_response_avg_diff.pt \\
        --steer-dialect Egyptian \\
        --layer 16 --coef 3.0
"""

import re
import torch
import argparse
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from activation_steer import ActivationSteerer


# ─── Dialect Evaluation Prompts ───────────────────────────────────────────────

MSA_EVAL_PROMPT = """I am evaluating a language model's Arabic output. I want to assess how formally the response is written — specifically whether it uses Modern Standard Arabic (الفصحى / MSA) rather than any colloquial dialect.

The model was asked the following question:
[QUESTION START]
{question}
[QUESTION END]

The model responded with:
[ANSWER START]
{answer}
[ANSWER END]

Rate how formally/MSA the response is on a scale of 0 to 100.
* 0 means "fully colloquial dialect, no MSA features."
* 50 means "mixed register."
* 100 means "fully formal MSA, no dialectal elements."

Your final output must be a single line containing only an integer score from 0 to 100, enclosed in `<score>` tags. For example: `<score>75</score>`. Do not add any other text.

<score>"""


def make_city_eval_prompt(city):
    """Return a dialect-authenticity evaluation prompt for any MADAR city name."""
    return f"""I am evaluating a language model's Arabic output. I want to assess how authentically the response sounds like the Arabic dialect spoken in {city}.

The model was asked the following question:
[QUESTION START]
{{question}}
[QUESTION END]

The model responded with:
[ANSWER START]
{{answer}}
[ANSWER END]

Rate how authentically the response sounds like the {city} Arabic dialect on a scale of 0 to 100.
* 0 means "no trace of {city} dialect; purely MSA or a completely different dialect."
* 50 means "some {city} dialect features but mixed or inconsistent."
* 100 means "fully authentic {city} dialect."

Focus only on dialect authenticity, not correctness or content quality.
Your final output must be a single line containing only an integer score from 0 to 100, enclosed in `<score>` tags. For example: `<score>75</score>`. Do not add any other text.

<score>"""

COHERENCE_PROMPT = """I am evaluating a language model. I want to assess the coherence of its answer — please focus solely on coherence.

The model was asked:
[QUESTION START]
{question}
[QUESTION END]

The model answered:
[ANSWER START]
{answer}
[ANSWER END]

Rate coherence on a scale of 0 to 100.
* 0 means "makes no sense; incoherent or gibberish."
* 50 means "partially understandable but rambling or confused."
* 100 means "clearly understood the question, well-formed response."

Your final output must be a single line containing only an integer score from 0 to 100, enclosed in `<score>` tags. For example: `<score>95</score>`. Do not add any explanation or other text.

<score>"""

ARABIC_FLUENCY_PROMPT = """I am evaluating a language model's Arabic output. I want to assess how fluent and natural the Arabic is.

The model was asked:
[QUESTION START]
{question}
[QUESTION END]

The model answered:
[ANSWER START]
{answer}
[ANSWER END]

Rate the naturalness and fluency of the Arabic on a scale of 0 to 100.
* 0 means "not Arabic at all, or completely broken/unnatural Arabic."
* 50 means "understandable but unnatural or awkward phrasing."
* 100 means "perfectly natural, fluent Arabic as a native speaker would produce."

Your final output must be a single line containing only an integer score from 0 to 100, enclosed in `<score>` tags. For example: `<score>80</score>`. Do not add any other text.

<score>"""


# ─── Utilities ────────────────────────────────────────────────────────────────

def parse_score(response_text):
    """Parse a score from <score>N</score> tags."""
    if "<score>REFUSAL</score>" in response_text:
        return None
    match = re.search(r"<score>(\d+)</score>", response_text)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None


def judge_with_model(model, tokenizer, eval_prompt, question, answer):
    """Use the loaded model as judge to score a response."""
    prompt = eval_prompt.format(question=question, answer=answer)
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        max_new_tokens=20,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=generation_config)

    input_len = inputs["input_ids"].shape[1]
    response_text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
    return parse_score(response_text)


def generate_response(model, tokenizer, prompt, max_new_tokens=300, temperature=0.7):
    """Generate a baseline response (no steering)."""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def generate_response_steered(model, tokenizer, prompt, vector, layer, coef,
                               max_new_tokens=300, temperature=0.7, steering_type="response"):
    """Generate a response with dialect activation steering applied."""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer - 1, positions=steering_type):
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


@contextmanager
def temporary_padding_side(tokenizer, padding_side):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        tokenizer.padding_side = original_padding_side


def _trim_generated_ids(tokenizer, token_ids):
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    trimmed = []
    for token_id in token_ids:
        if token_id in special_ids:
            break
        trimmed.append(token_id)
    return trimmed


def _top_logprob_dicts(tokenizer, step_logprobs, topk):
    top_values, top_indices = torch.topk(
        step_logprobs, k=min(topk, step_logprobs.shape[-1]), dim=-1
    )
    results = []
    for row_values, row_indices in zip(top_values, top_indices):
        results.append(
            {
                tokenizer.decode([idx.item()], skip_special_tokens=False): value.item()
                for idx, value in zip(row_indices, row_values)
            }
        )
    return results


def generate_responses_steered_batched(model, tokenizer, prompts, vector, layer, coef,
                                       max_new_tokens=300, temperature=0.7, steering_type="response",
                                       return_logprobs=False, top_logprobs=5):
    """Generate steered responses for a batch of prompts in a single model pass."""
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
    prompt_texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    with temporary_padding_side(tokenizer, "left"):
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)

    prompt_width = inputs["input_ids"].shape[1]

    with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer - 1, positions=steering_type):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=return_logprobs,
                output_scores=return_logprobs,
            )

    if not return_logprobs:
        sequences = outputs
        return [
            tokenizer.decode(sequences[row_idx, prompt_width:], skip_special_tokens=True)
            for row_idx in range(len(prompts))
        ]

    sequences = outputs.sequences
    generated_token_matrix = sequences[:, prompt_width:]
    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        getattr(outputs, "beam_indices", None),
        normalize_logits=True,
    )
    generated_transition_scores = transition_scores[:, -generated_token_matrix.shape[1]:]
    topk = max(int(top_logprobs), 1)
    top_logprobs_per_step = []
    for step_scores in outputs.scores:
        step_logprobs = torch.log_softmax(step_scores, dim=-1)
        top_logprobs_per_step.append(_top_logprob_dicts(tokenizer, step_logprobs, topk))

    results = []
    for row_idx in range(len(prompts)):
        token_ids = _trim_generated_ids(tokenizer, generated_token_matrix[row_idx].detach().cpu().tolist())
        actual_len = len(token_ids)
        token_texts = [
            tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids
        ]
        token_logprobs = (
            generated_transition_scores[row_idx, :actual_len].detach().cpu().tolist()
            if actual_len
            else []
        )
        row_top_logprobs = [
            top_logprobs_per_step[step_idx][row_idx]
            for step_idx in range(actual_len)
        ]
        results.append(
            {
                "text": tokenizer.decode(token_ids, skip_special_tokens=True),
                "token_ids": token_ids,
                "tokens": token_texts,
                "token_logprobs": token_logprobs,
                "top_logprobs": row_top_logprobs,
            }
        )
    return results


def _score_prompts_teacher_forced_batched(model, tokenizer, prompts, vector, layer, coef,
                                          steering_positions, topk):
    with temporary_padding_side(tokenizer, "right"):
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(model.device)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    lengths = attention_mask.sum(dim=1).tolist()

    with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer - 1, positions=steering_positions):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logprobs = torch.log_softmax(outputs.logits, dim=-1)

    results = []
    for row_idx, prompt in enumerate(prompts):
        length = int(lengths[row_idx])
        if length == 0:
            results.append(
                {
                    "text": "",
                    "token_ids": [],
                    "tokens": [],
                    "token_logprobs": [],
                    "top_logprobs": [],
                    "text_offset": [],
                    "next_token": None,
                    "next_token_logprob": None,
                    "next_top_logprobs": None,
                }
            )
            continue

        token_ids = input_ids[row_idx, :length].detach().cpu().tolist()
        tokens = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in token_ids]
        text_offsets = []
        offset = 0
        for token in tokens:
            text_offsets.append(offset)
            offset += len(token)

        token_logprobs = [None]
        top_logprobs_list = [None]
        for pos in range(1, length):
            step_logprobs = all_logprobs[row_idx, pos - 1]
            target_id = token_ids[pos]
            token_logprobs.append(step_logprobs[target_id].item())
            top_logprobs_list.append(_top_logprob_dicts(tokenizer, step_logprobs.unsqueeze(0), topk)[0])

        final_logprobs = all_logprobs[row_idx, length - 1]
        next_token_id = torch.argmax(final_logprobs).item()
        next_top_logprobs = _top_logprob_dicts(tokenizer, final_logprobs.unsqueeze(0), topk)[0]

        results.append(
            {
                "text": prompt,
                "token_ids": token_ids,
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs_list,
                "text_offset": text_offsets,
                "next_token": tokenizer.decode([next_token_id], skip_special_tokens=False),
                "next_token_logprob": final_logprobs[next_token_id].item(),
                "next_top_logprobs": next_top_logprobs,
            }
        )
    return results


def _select_past_key_values(past_key_values, selected_rows):
    index = torch.tensor(selected_rows, device=past_key_values[0][0].device, dtype=torch.long)
    selected = []
    for layer_past in past_key_values:
        selected.append(tuple(past.index_select(0, index) for past in layer_past))
    return tuple(selected)


def _score_prompts_response_exact_batched(model, tokenizer, prompts, vector, layer, coef, topk):
    tokenized = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    batch_size = len(prompts)
    results = []
    for prompt, token_ids in zip(prompts, tokenized):
        tokens = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in token_ids]
        text_offsets = []
        offset = 0
        for token in tokens:
            text_offsets.append(offset)
            offset += len(token)
        results.append(
            {
                "text": prompt,
                "token_ids": list(token_ids),
                "tokens": tokens,
                "token_logprobs": [None] if token_ids else [],
                "top_logprobs": [None] if token_ids else [],
                "text_offset": text_offsets,
                "next_token": None,
                "next_token_logprob": None,
                "next_top_logprobs": None,
            }
        )

    active_indices = [idx for idx, token_ids in enumerate(tokenized) if token_ids]
    if not active_indices:
        return results

    current_positions = {idx: 0 for idx in active_indices}
    current_tokens = torch.tensor(
        [[tokenized[idx][0]] for idx in active_indices],
        device=model.device,
        dtype=torch.long,
    )
    past_key_values = None

    with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer - 1, positions="response"):
        with torch.no_grad():
            while active_indices:
                outputs = model(
                    input_ids=current_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                step_logprobs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
                next_active_indices = []
                next_tokens = []
                next_selected_rows = []

                for row_idx, prompt_idx in enumerate(active_indices):
                    token_ids = tokenized[prompt_idx]
                    pos = current_positions[prompt_idx]
                    result = results[prompt_idx]

                    if pos == len(token_ids) - 1:
                        next_token_id = torch.argmax(step_logprobs[row_idx]).item()
                        result["next_token"] = tokenizer.decode([next_token_id], skip_special_tokens=False)
                        result["next_token_logprob"] = step_logprobs[row_idx, next_token_id].item()
                        result["next_top_logprobs"] = _top_logprob_dicts(
                            tokenizer, step_logprobs[row_idx].unsqueeze(0), topk
                        )[0]

                    if pos + 1 < len(token_ids):
                        target_id = token_ids[pos + 1]
                        result["token_logprobs"].append(step_logprobs[row_idx, target_id].item())
                        result["top_logprobs"].append(
                            _top_logprob_dicts(
                                tokenizer, step_logprobs[row_idx].unsqueeze(0), topk
                            )[0]
                        )
                        current_positions[prompt_idx] = pos + 1
                        next_active_indices.append(prompt_idx)
                        next_tokens.append([target_id])
                        next_selected_rows.append(row_idx)

                if not next_active_indices:
                    break

                current_tokens = torch.tensor(
                    next_tokens,
                    device=model.device,
                    dtype=torch.long,
                )
                past_key_values = _select_past_key_values(outputs.past_key_values, next_selected_rows)
                active_indices = next_active_indices

    return results


def score_prompts_steered_batched(model, tokenizer, prompts, vector, layer, coef,
                                  steering_type="response", top_logprobs=5,
                                  response_scoring_mode="fast"):
    """Score a batch of prompts under steering for completion-style logprob requests."""
    topk = max(int(top_logprobs), 1)
    if steering_type == "response" and response_scoring_mode == "exact":
        return _score_prompts_response_exact_batched(
            model, tokenizer, prompts, vector, layer, coef, topk
        )

    steering_positions = steering_type
    if steering_type == "response" and response_scoring_mode == "fast":
        steering_positions = "all"

    return _score_prompts_teacher_forced_batched(
        model, tokenizer, prompts, vector, layer, coef, steering_positions, topk
    )


def score_prompt_steered(model, tokenizer, prompt, vector, layer, coef,
                         steering_type="response", top_logprobs=5,
                         response_scoring_mode="fast"):
    """Single-prompt wrapper around the batched scoring helper."""
    return score_prompts_steered_batched(
        model,
        tokenizer,
        [prompt],
        vector,
        layer,
        coef,
        steering_type=steering_type,
        top_logprobs=top_logprobs,
        response_scoring_mode=response_scoring_mode,
    )[0]


def evaluate_response(model, tokenizer, question, answer, eval_metrics):
    """Evaluate a response on all metrics. Returns {metric_name: score}."""
    scores = {}
    for metric_name, eval_prompt in eval_metrics.items():
        scores[metric_name] = judge_with_model(model, tokenizer, eval_prompt, question, answer)
    return scores


# ─── Output ───────────────────────────────────────────────────────────────────

def print_comparison(prompt, baseline_response, steered_response,
                     baseline_scores, steered_scores, steer_dialect, coef, layer):
    """Print a formatted before/after comparison."""

    print(f"\n{'='*80}")
    print("DIALECT STEERING COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"Target dialect: {steer_dialect} | Coefficient: {coef} | Layer: {layer}")
    print(f"{'='*80}")

    print(f"\n{'─'*80}")
    print("PROMPT:")
    print(f"{'─'*80}")
    print(prompt)

    print(f"\n{'─'*80}")
    print("BASELINE RESPONSE (No Steering):")
    print(f"{'─'*80}")
    print(baseline_response)

    print(f"\n{'─'*80}")
    print(f"STEERED RESPONSE (coef={coef}, dialect={steer_dialect}):")
    print(f"{'─'*80}")
    print(steered_response)

    print(f"\n{'='*80}")
    print("EVALUATION SCORES (0–100)")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Baseline':>10} {'Steered':>10} {'Delta':>10} {'Direction':>12}")
    print(f"{'─'*70}")

    for metric in baseline_scores:
        b = baseline_scores[metric]
        s = steered_scores[metric]

        b_str = f"{b:.1f}" if b is not None else "N/A"
        s_str = f"{s:.1f}" if s is not None else "N/A"

        if b is not None and s is not None:
            delta = s - b
            delta_str = f"{delta:+.1f}"
            direction = "~" if abs(delta) < 5 else ("UP" if delta > 0 else "DOWN")
        else:
            delta_str, direction = "N/A", "?"

        marker = " <-- TARGET" if metric == steer_dialect else ""
        print(f"{metric:<25} {b_str:>10} {s_str:>10} {delta_str:>10} {direction:>12}{marker}")

    print(f"{'─'*70}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if steer_dialect in baseline_scores and steer_dialect in steered_scores:
        b = baseline_scores[steer_dialect]
        s = steered_scores[steer_dialect]
        if b is not None and s is not None:
            delta = s - b
            if delta > 10:
                print(f"Steering INCREASED {steer_dialect} dialect score by {delta:.1f} points")
            elif delta < -10:
                print(f"Steering DECREASED {steer_dialect} dialect score by {abs(delta):.1f} points")
            else:
                print(f"Steering had MINIMAL effect on {steer_dialect} score ({delta:+.1f} points)")

    if "coherence" in baseline_scores and "coherence" in steered_scores:
        b_coh = baseline_scores["coherence"]
        s_coh = steered_scores["coherence"]
        if b_coh is not None and s_coh is not None:
            coh_delta = s_coh - b_coh
            if coh_delta < -15:
                print(f"WARNING: Steering degraded coherence by {abs(coh_delta):.1f} points")
            else:
                print(f"Coherence remained stable ({coh_delta:+.1f} points)")

    print(f"{'='*80}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Steer a model towards an Arabic dialect and compare before/after",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python steer_dialect_and_compare.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --vector-path dialect_vectors/Qwen2.5-7B-Instruct/Levantine_response_avg_diff.pt \\
      --steer-dialect Levantine \\
      --layer 16 --coef 3.0 \\
      --prompt "كيف حالك اليوم؟"

  # Evaluate on multiple dialects simultaneously:
  python steer_dialect_and_compare.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --vector-path dialect_vectors/Qwen2.5-7B-Instruct/Egyptian_response_avg_diff.pt \\
      --steer-dialect Egyptian \\
      --layer 16 --coef 4.0 \\
      --eval-dialects Egyptian Levantine MSA coherence arabic_fluency
        """
    )

    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--vector-path", type=str, required=True,
                        help="Path to the dialect vector (.pt file)")
    parser.add_argument("--steer-dialect", type=str, required=True,
                        help="Name of the dialect being steered (e.g. Levantine)")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer index to apply steering")
    parser.add_argument("--coef", type=float, default=3.0,
                        help="Steering coefficient (default: 3.0)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Arabic prompt to evaluate (if omitted, will ask interactively)")
    parser.add_argument("--eval-dialects", nargs="+",
                        default=None,
                        help="Metrics to evaluate on. Options: Levantine Egyptian Gulf Moroccan MSA coherence arabic_fluency. "
                             "Defaults to the steering dialect + coherence + arabic_fluency.")
    parser.add_argument("--steering-type", type=str, default="response",
                        choices=["all", "prompt", "response"],
                        help="Where to apply steering (default: response)")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Max tokens to generate (default: 300)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")

    args = parser.parse_args()

    # Default eval metrics
    if args.eval_dialects is None:
        args.eval_dialects = [args.steer_dialect, "coherence", "arabic_fluency"]

    # ─── Print Config ──────────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("ARABIC DIALECT STEERING COMPARISON")
    print(f"{'='*80}")
    print(f"Model:           {args.model}")
    print(f"Target dialect:  {args.steer_dialect}")
    print(f"Vector path:     {args.vector_path}")
    print(f"Layer:           {args.layer}")
    print(f"Coefficient:     {args.coef}")
    print(f"Steering type:   {args.steering_type}")
    print(f"Eval metrics:    {', '.join(args.eval_dialects)}")
    print(f"{'='*80}\n")

    # ─── Load Model ────────────────────────────────────────────────────────────

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

    device = next(model.parameters()).device
    print(f"Model loaded on: {device}")

    # ─── Load Steering Vector ──────────────────────────────────────────────────

    print(f"\nLoading dialect vector from: {args.vector_path}")
    all_layers = torch.load(args.vector_path, weights_only=False)
    vector = all_layers[args.layer]
    print(f"Vector shape: {vector.shape}  (from layer {args.layer})")

    # ─── Build Eval Metrics ────────────────────────────────────────────────────

    print(f"\nBuilding evaluation prompts for: {args.eval_dialects}")
    eval_metrics = {}
    for metric in args.eval_dialects:
        if metric == "coherence":
            eval_metrics["coherence"] = COHERENCE_PROMPT
            print(f"  Loaded: coherence (built-in)")
        elif metric == "arabic_fluency":
            eval_metrics["arabic_fluency"] = ARABIC_FLUENCY_PROMPT
            print(f"  Loaded: arabic_fluency (built-in)")
        elif metric == "MSA":
            eval_metrics["MSA"] = MSA_EVAL_PROMPT
            print(f"  Loaded: MSA formality evaluator (built-in)")
        else:
            # Treat any other string as a city name and generate a prompt dynamically
            eval_metrics[metric] = make_city_eval_prompt(metric)
            print(f"  Loaded: {metric} dialect evaluator (city-based)")

    if not eval_metrics:
        print("Error: no evaluation metrics loaded. Exiting.")
        return

    # ─── Get Prompt ────────────────────────────────────────────────────────────

    if args.prompt:
        prompt = args.prompt
    else:
        print(f"\n{'─'*80}")
        prompt = input("Enter your Arabic prompt: ").strip()
        if not prompt:
            print("Error: empty prompt. Exiting.")
            return

    print(f"\nPrompt: {prompt}")

    # ─── Step 1: Baseline ─────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("STEP 1: Generating BASELINE response (no steering)...")
    print(f"{'='*80}")

    baseline_response = generate_response(
        model, tokenizer, prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"\nBaseline:\n{baseline_response}")

    # ─── Step 2: Evaluate Baseline ─────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("STEP 2: Evaluating BASELINE response...")
    print(f"{'='*80}")

    baseline_scores = evaluate_response(model, tokenizer, prompt, baseline_response, eval_metrics)
    for metric, score in baseline_scores.items():
        print(f"  {metric:<25} {f'{score:.1f}' if score is not None else 'N/A'}")

    # ─── Step 3: Steered Response ─────────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"STEP 3: Generating STEERED response (dialect={args.steer_dialect}, coef={args.coef})...")
    print(f"{'='*80}")

    steered_response = generate_response_steered(
        model, tokenizer, prompt, vector, args.layer, args.coef,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        steering_type=args.steering_type,
    )
    print(f"\nSteered:\n{steered_response}")

    # ─── Step 4: Evaluate Steered ─────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("STEP 4: Evaluating STEERED response...")
    print(f"{'='*80}")

    steered_scores = evaluate_response(model, tokenizer, prompt, steered_response, eval_metrics)
    for metric, score in steered_scores.items():
        print(f"  {metric:<25} {f'{score:.1f}' if score is not None else 'N/A'}")

    # ─── Step 5: Comparison ────────────────────────────────────────────────────

    print_comparison(
        prompt, baseline_response, steered_response,
        baseline_scores, steered_scores,
        args.steer_dialect, args.coef, args.layer,
    )


if __name__ == "__main__":
    main()
