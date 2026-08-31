#!/usr/bin/env python3
"""
OpenAI-style local completions server for lm-eval using activation steering.

This server wraps the existing `generate_response_steered` function from
`steer_dialect_and_compare.py` so it can be used through lm-evaluation-harness'
API model adapters.

What this supports:
- `generate_until` style tasks through `/v1/completions`
- optional remote tokenizer endpoints: `/tokenizer_info` and `/tokenize`

What this does not support:
- loglikelihood / MCQ tasks that require token logprobs

Example:
    python arabic_steering_vector/lm_eval_steered_api.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --vector-path arabic_steering_vector/dialect_vectors/Qwen2.5-7B-Instruct/Cairo_response_avg_diff.pt \
        --layer 16 \
        --coef 3.0 \
        --served-model-name qwen-steered-cairo

Then point lm_eval at:
    --model local-completions
    --model_args model=qwen-steered-cairo,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=Qwen/Qwen2.5-7B-Instruct,tokenizer_backend=huggingface,tokenized_requests=False

Recommended:
- use generation-based tasks
- do not add `--apply_chat_template`, since this server already wraps prompts as
  a single user message before calling `generate_response_steered`
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

@dataclass
class ServerState:
    model_name: str
    tokenizer_name: str
    torch_module: Any
    model: Any
    tokenizer: Any
    vector: Any
    generate_batch_fn: Any
    score_prompts_fn: Any
    layer: int
    coef: float
    steering_type: str
    echo_scoring_mode: str
    default_max_tokens: int
    default_temperature: float
    generation_lock: threading.Lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a steered model behind an OpenAI-style completions API for lm_eval."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer name/path. Defaults to the model name.",
    )
    parser.add_argument(
        "--vector-path",
        required=True,
        help="Path to the steering vector .pt file",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="1-based layer index, matching steer_dialect_and_compare.py",
    )
    parser.add_argument(
        "--coef",
        type=float,
        default=3.0,
        help="Steering coefficient",
    )
    parser.add_argument(
        "--steering-type",
        choices=["all", "prompt", "response"],
        default="response",
        help="Where steering is applied",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model name exposed by the API. Defaults to '<base>-steered'.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Default max tokens when the request omits max_tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Default temperature when the request omits temperature",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype for model loading",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Passed to from_pretrained. Use 'auto', a device_map string, or 'none'.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to model/tokenizer loading",
    )
    parser.add_argument(
        "--echo-scoring-mode",
        choices=["fast", "exact"],
        default="fast",
        help="How to score echo=true prompt logprobs for response steering.",
    )
    return parser


def get_runtime_imports() -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steer_dialect_and_compare import (
        generate_responses_steered_batched,
        score_prompts_steered_batched,
    )

    return (
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        generate_responses_steered_batched,
        score_prompts_steered_batched,
    )


def resolve_dtype(dtype_name: str) -> Any:
    torch, _, _, _, _ = get_runtime_imports()
    if dtype_name == "auto":
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_state(args: argparse.Namespace) -> ServerState:
    torch, AutoModelForCausalLM, AutoTokenizer, generate_responses_steered_batched, score_prompts_steered_batched = (
        get_runtime_imports()
    )
    tokenizer_name = args.tokenizer or args.model
    served_model_name = args.served_model_name or f"{Path(args.model).name}-steered"

    load_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    torch_dtype = resolve_dtype(args.dtype)
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map

    print(f"Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    print(f"Loading tokenizer: {tokenizer_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading steering vector: {args.vector_path}", flush=True)
    all_layers = torch.load(args.vector_path, weights_only=False)
    vector = all_layers[args.layer]

    return ServerState(
        model_name=served_model_name,
        tokenizer_name=tokenizer_name,
        torch_module=torch,
        model=model,
        tokenizer=tokenizer,
        vector=vector,
        generate_batch_fn=generate_responses_steered_batched,
        score_prompts_fn=score_prompts_steered_batched,
        layer=args.layer,
        coef=args.coef,
        steering_type=args.steering_type,
        echo_scoring_mode=args.echo_scoring_mode,
        default_max_tokens=args.max_tokens,
        default_temperature=args.temperature,
        generation_lock=threading.Lock(),
    )


def decode_prompt(tokenizer: Any, prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and all(isinstance(tok, int) for tok in prompt):
        return tokenizer.decode(prompt, skip_special_tokens=False)
    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def normalize_prompts(tokenizer: Any, prompt_field: Any) -> list[str]:
    if isinstance(prompt_field, str):
        return [prompt_field]
    if isinstance(prompt_field, list):
        if not prompt_field:
            return [""]
        if all(isinstance(item, str) for item in prompt_field):
            return list(prompt_field)
        if all(isinstance(item, int) for item in prompt_field):
            return [decode_prompt(tokenizer, prompt_field)]
        if all(isinstance(item, list) for item in prompt_field):
            return [decode_prompt(tokenizer, item) for item in prompt_field]
    raise TypeError("`prompt` must be a string, list[str], list[int], or list[list[int]]")


def apply_stop_sequences(text: str, stop: Any) -> tuple[str, str]:
    if stop is None:
        return text, "length"
    stop_list = stop if isinstance(stop, list) else [stop]
    stop_list = [item for item in stop_list if item]
    if not stop_list:
        return text, "length"

    earliest = None
    for marker in stop_list:
        idx = text.find(marker)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return text, "length"
    return text[:earliest], "stop"


def estimate_usage(tokenizer: Any, prompt: str, completion: str) -> dict[str, int]:
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_error(message: str, err_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type}}


def parse_logprobs_from_generation(prompt: str, generation: dict[str, Any]) -> dict[str, Any]:
    tokens = generation.get("tokens", [])
    token_logprobs = generation.get("token_logprobs", [])
    top_logprobs = generation.get("top_logprobs", [])
    text_offsets = []
    offset = len(prompt)
    for token in tokens:
        text_offsets.append(offset)
        offset += len(token)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offsets,
    }


def parse_echo_logprobs_from_prompt(score_data: dict[str, Any], include_next_token: bool) -> dict[str, Any]:
    tokens = list(score_data.get("tokens", []))
    token_logprobs = list(score_data.get("token_logprobs", []))
    top_logprobs = list(score_data.get("top_logprobs", []))
    text_offsets = list(score_data.get("text_offset", []))

    if include_next_token and score_data.get("next_token") is not None:
        next_offset = text_offsets[-1] + len(tokens[-1]) if tokens else 0
        tokens.append(score_data["next_token"])
        token_logprobs.append(score_data["next_token_logprob"])
        top_logprobs.append(score_data["next_top_logprobs"])
        text_offsets.append(next_offset)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offsets,
    }


def make_handler(state: ServerState):
    class SteeredCompletionHandler(BaseHTTPRequestHandler):
        server_version = "SteeredLMEvalAPI/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(
                "%s - - [%s] %s\n"
                % (self.address_string(), self.log_date_time_string(), fmt % args)
            )

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json({"status": "ok", "model": state.model_name})
                return
            if self.path == "/v1/models":
                self._write_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": state.model_name,
                                "object": "model",
                                "owned_by": "local",
                            }
                        ],
                    }
                )
                return
            if self.path == "/tokenizer_info":
                self._write_json(
                    {
                        "eos_token": state.tokenizer.eos_token,
                        "bos_token": state.tokenizer.bos_token,
                        "pad_token": state.tokenizer.pad_token,
                        "model_max_length": getattr(
                            state.tokenizer, "model_max_length", None
                        ),
                        "chat_template": getattr(state.tokenizer, "chat_template", None),
                    }
                )
                return
            self._write_json(build_error("Not found", "not_found_error"), status=404)

        def do_POST(self) -> None:
            try:
                if self.path == "/tokenize":
                    self._handle_tokenize()
                    return
                if self.path == "/v1/completions":
                    self._handle_completions()
                    return
                self._write_json(build_error("Not found", "not_found_error"), status=404)
            except Exception as exc:
                self._write_json(
                    build_error(f"Server error: {exc}", "server_error"),
                    status=500,
                )

        def _handle_tokenize(self) -> None:
            payload = self._read_json()
            prompt = payload.get("prompt", "")
            add_special_tokens = bool(payload.get("add_special_tokens", False))
            if not isinstance(prompt, str):
                self._write_json(
                    build_error("`prompt` must be a string for /tokenize"),
                    status=400,
                )
                return
            tokens = state.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
            self._write_json({"tokens": tokens})

        def _handle_completions(self) -> None:
            payload = self._read_json()
            prompt_field = payload.get("prompt")
            if prompt_field is None:
                self._write_json(build_error("Missing required field: prompt"), status=400)
                return

            prompts = normalize_prompts(state.tokenizer, prompt_field)
            requested_model = payload.get("model", state.model_name)
            max_tokens = int(payload.get("max_tokens", state.default_max_tokens))
            temperature = float(payload.get("temperature", state.default_temperature))
            stop = payload.get("stop")
            requested_logprobs = payload.get("logprobs")
            seed = payload.get("seed")
            echo_prompt = bool(payload.get("echo", False))

            if seed is not None:
                state.torch_module.manual_seed(int(seed))
                if state.torch_module.cuda.is_available():
                    state.torch_module.cuda.manual_seed_all(int(seed))

            choices = []
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            if echo_prompt:
                with state.generation_lock:
                    scored_prompts = state.score_prompts_fn(
                        state.model,
                        state.tokenizer,
                        prompts,
                        state.vector,
                        state.layer,
                        state.coef,
                        steering_type=state.steering_type,
                        top_logprobs=requested_logprobs or 5,
                        response_scoring_mode=state.echo_scoring_mode,
                    )

                for index, (prompt, scored_prompt) in enumerate(zip(prompts, scored_prompts)):
                    appended_text = ""
                    include_next_token = max_tokens > 0
                    if include_next_token and scored_prompt.get("next_token") is not None:
                        appended_text = scored_prompt["next_token"]
                    full_text = prompt + appended_text
                    token_usage = estimate_usage(state.tokenizer, prompt, appended_text)
                    for key in usage:
                        usage[key] += token_usage[key]

                    choices.append(
                        {
                            "text": full_text,
                            "index": index,
                            "logprobs": parse_echo_logprobs_from_prompt(
                                scored_prompt,
                                include_next_token=include_next_token,
                            ),
                            "finish_reason": "length" if include_next_token else "stop",
                        }
                    )
            else:
                with state.generation_lock:
                    generated_batch = state.generate_batch_fn(
                        state.model,
                        state.tokenizer,
                        prompts,
                        state.vector,
                        state.layer,
                        state.coef,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        steering_type=state.steering_type,
                        return_logprobs=requested_logprobs is not None,
                        top_logprobs=requested_logprobs or 5,
                    )

                for index, (prompt, generated) in enumerate(zip(prompts, generated_batch)):
                    if requested_logprobs is not None:
                        full_text = generated["text"]
                    else:
                        full_text = generated

                    trimmed, finish_reason = apply_stop_sequences(full_text, stop)
                    token_usage = estimate_usage(state.tokenizer, prompt, trimmed)
                    for key in usage:
                        usage[key] += token_usage[key]

                    choice_logprobs = None
                    if requested_logprobs is not None:
                        if trimmed != full_text:
                            trim_at = len(trimmed)
                            kept_chars = 0
                            kept_tokens = 0
                            for token in generated["tokens"]:
                                next_chars = kept_chars + len(token)
                                if next_chars > trim_at:
                                    break
                                kept_chars = next_chars
                                kept_tokens += 1
                            generated = {
                                **generated,
                                "tokens": generated["tokens"][:kept_tokens],
                                "token_ids": generated["token_ids"][:kept_tokens],
                                "token_logprobs": generated["token_logprobs"][:kept_tokens],
                                "top_logprobs": generated["top_logprobs"][:kept_tokens],
                            }
                        choice_logprobs = parse_logprobs_from_generation(prompt, generated)

                    choices.append(
                        {
                            "text": trimmed,
                            "index": index,
                            "logprobs": choice_logprobs,
                            "finish_reason": finish_reason,
                        }
                    )

            self._write_json(
                {
                    "id": f"cmpl-{uuid.uuid4().hex}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": requested_model,
                    "choices": choices,
                    "usage": usage,
                }
            )

    return SteeredCompletionHandler


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    state = load_state(args)

    handler_cls = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    print("", flush=True)
    print(f"Server ready at http://{args.host}:{args.port}", flush=True)
    print(f"Served model name: {state.model_name}", flush=True)
    print(f"Tokenizer name: {state.tokenizer_name}", flush=True)
    print("", flush=True)
    print("lm_eval example:", flush=True)
    print(
        "lm_eval "
        "--model local-completions "
        f"--model_args model={state.model_name},base_url=http://{args.host}:{args.port}/v1/completions,"
        f"tokenizer={state.tokenizer_name},tokenizer_backend=huggingface,tokenized_requests=False "
        "--tasks <generation_task>",
        flush=True,
    )
    print("", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
