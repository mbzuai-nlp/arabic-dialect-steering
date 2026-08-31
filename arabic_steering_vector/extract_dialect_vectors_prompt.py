"""
Extract Arabic dialect vectors from the PROMPT (user-turn) position.

Like extract_dialect_vectors_fast.py but places each MADAR sentence in the
USER turn instead of the assistant turn, then extracts hidden states from
those prompt token positions only.  The resulting vectors are principled
for use with --steering-type prompt in generate_steered_responses.py.

Saved as: dialect_vectors/{model}/{City}_prompt_avg_diff.pt
  (shape: [num_layers, hidden_dim])

Usage:
    python extract_dialect_vectors_prompt.py \\
        --model QCRI/Fanar-1-9B-Instruct --dialects Cairo Rabat

    python extract_dialect_vectors_prompt.py \\
        --model humain-ai/ALLaM-7B-Instruct-preview --all-dialects
"""

import os
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


MSA_TSV_FILENAME = "MADAR.corpus.MSA.tsv"
PLACEHOLDER = "XXXXXXXXXXX"   # used to locate text boundaries in the template


# ─── Data ─────────────────────────────────────────────────────────────────────

def discover_cities(data_dir="."):
    tsv_files = Path(data_dir).glob("MADAR.corpus.*.tsv")
    cities = []
    for f in sorted(tsv_files):
        city = f.stem.replace("MADAR.corpus.", "")
        if city != "MSA":
            cities.append(city)
    return cities


def load_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype={"sentID.BTEC": int})
    df = df.rename(columns={"sentID.BTEC": "sentID", "sent": "text"})
    return df.set_index("sentID")


def load_parallel_samples(city, n=None, data_dir="."):
    city_path = Path(data_dir) / f"MADAR.corpus.{city}.tsv"
    msa_path  = Path(data_dir) / MSA_TSV_FILENAME

    if not city_path.exists():
        raise FileNotFoundError(f"City TSV not found: {city_path}")
    if not msa_path.exists():
        raise FileNotFoundError(f"MSA TSV not found: {msa_path}")

    city_df = load_tsv(city_path)
    msa_df  = load_tsv(msa_path)

    common_ids = sorted(city_df.index.intersection(msa_df.index))

    if n is None:
        n = len(common_ids)
    elif n > len(common_ids):
        print(f"  Warning: only {len(common_ids)} parallel sentences for '{city}', using all.")
        n = len(common_ids)

    selected_ids = common_ids[:n]
    return (
        city_df.loc[selected_ids, "text"].tolist(),
        msa_df.loc[selected_ids, "text"].tolist(),
        selected_ids,
    )


# ─── Hidden State Extraction ──────────────────────────────────────────────────

def _get_text_slice(tokenizer, prefix_str, text):
    """
    Return (start, end) token indices of `text` within the formatted prompt.
    Uses a placeholder to locate the text boundary robustly.
    """
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    text_ids   = tokenizer.encode(text,       add_special_tokens=False)
    start = len(prefix_ids)
    end   = start + len(text_ids)
    return start, end


def get_prompt_hidden_states(model, tokenizer, texts, prefix_str, suffix_str, layer_list=None):
    """
    Extract hidden states from the dialect text tokens only (user-turn position).

    Args:
        texts:      List of dialect/MSA sentences placed in the user turn.
        prefix_str: Chat-template string that precedes the text (e.g. '<|user|>').
        suffix_str: Chat-template string that follows the text (e.g. '<|end|>\\n<|assistant|>\\n').
        layer_list: Which layers to extract (default: all).

    Returns:
        List of tensors, one per layer, shape [n_samples, hidden_dim].
    """
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer + 1))

    layer_avgs = [[] for _ in range(max_layer + 1)]

    for text in tqdm(texts, desc="Extracting hidden states"):
        full_str = prefix_str + text + suffix_str
        start, end = _get_text_slice(tokenizer, prefix_str, text)

        inputs = tokenizer(full_str, return_tensors="pt", add_special_tokens=False).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer in layer_list:
            # Average over the dialect text token positions only
            layer_avgs[layer].append(
                outputs.hidden_states[layer][:, start:end, :].mean(dim=1).detach().cpu()
            )

        del outputs

    for layer in layer_list:
        layer_avgs[layer] = torch.cat(layer_avgs[layer], dim=0)

    return layer_avgs


def build_template_parts(tokenizer):
    """
    Use a placeholder to split the chat template into the prefix and suffix
    that surround the user-turn text.

    Returns:
        prefix_str: everything before the dialect text
        suffix_str: everything after the dialect text (closing tag + assistant prompt)
    """
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": PLACEHOLDER}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if PLACEHOLDER not in template:
        raise ValueError(
            f"Placeholder '{PLACEHOLDER}' not found in chat template output. "
            "Choose a placeholder that won't appear in the template itself."
        )
    prefix_str, suffix_str = template.split(PLACEHOLDER, maxsplit=1)
    return prefix_str, suffix_str


# ─── Vector Extraction ────────────────────────────────────────────────────────

def extract_dialect_vector(model, tokenizer, city, prefix_str, suffix_str,
                           n_samples=None, data_dir="."):
    print(f"\n{'='*60}")
    print(f"Extracting PROMPT-side vector for city: {city}")
    print(f"{'='*60}")

    pos_texts, neg_texts, sent_ids = load_parallel_samples(city, n_samples, data_dir)

    print(f"  Parallel sentences used: {len(sent_ids)}")
    print(f"  Positive ({city}) | Negative (MSA) — first 3 pairs:")
    for i in range(min(3, len(sent_ids))):
        print(f"    [{sent_ids[i]}] + {pos_texts[i]}")
        print(f"    [{sent_ids[i]}] - {neg_texts[i]}")
        print()

    print(f"  Extracting positive ({city}) hidden states...")
    pos_avg = get_prompt_hidden_states(model, tokenizer, pos_texts, prefix_str, suffix_str)

    print(f"  Extracting negative (MSA) hidden states...")
    neg_avg = get_prompt_hidden_states(model, tokenizer, neg_texts, prefix_str, suffix_str)

    print(f"\n  Computing diff across {len(pos_avg)} layers...")
    prompt_avg_diff = torch.stack([
        pos_avg[l].mean(0).float() - neg_avg[l].mean(0).float()
        for l in range(len(pos_avg))
    ], dim=0)

    return prompt_avg_diff


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_dialect_vectors(vectors_dict, output_dir, layer_idx=16):
    os.makedirs(output_dir, exist_ok=True)

    dialects = list(vectors_dict.keys())
    print(f"\n{'='*60}")
    print(f"Visualizing {len(dialects)} dialects at layer {layer_idx}")
    print(f"{'='*60}")

    layer_vectors = {}
    for dialect, vector in vectors_dict.items():
        if vector.shape[0] > layer_idx:
            layer_vectors[dialect] = vector[layer_idx]
        else:
            print(f"  Warning: '{dialect}' has fewer than {layer_idx+1} layers, skipping.")

    if len(layer_vectors) < 2:
        print("  Need at least 2 vectors for analysis.")
        return

    dialects = list(layer_vectors.keys())

    # Cosine similarity heatmap
    normalized = {d: v / torch.linalg.norm(v) for d, v in layer_vectors.items()}
    sim_matrix = pd.DataFrame(index=dialects, columns=dialects, dtype=float)
    for d1 in dialects:
        for d2 in dialects:
            sim_matrix.loc[d1, d2] = torch.dot(normalized[d1], normalized[d2]).item()

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, cmap="viridis", fmt=".2f",
                cbar_kws={"label": "Cosine Similarity"})
    plt.title(f"Dialect Vector Cosine Similarity — Prompt-side (Layer {layer_idx})",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"prompt_similarity_heatmap_layer{layer_idx}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {heatmap_path}")

    # PCA 2D
    vector_matrix = torch.stack([layer_vectors[d] for d in dialects]).numpy()
    n_components = min(5, len(dialects))
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(vector_matrix)

    pca_df = pd.DataFrame(pcs[:, :2], index=dialects, columns=["PC1", "PC2"])
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=300,
                    color="steelblue", edgecolor="black", linewidth=1.5)
    plt.title(f"Dialect Vectors PCA — Prompt-side (Layer {layer_idx})",
              fontsize=14, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=12)
    plt.grid(True, alpha=0.3)
    for i, d in enumerate(dialects):
        plt.annotate(d, (pca_df["PC1"].iloc[i], pca_df["PC2"].iloc[i]),
                     xytext=(5, 5), textcoords="offset points", fontsize=10,
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    plt.tight_layout()
    pca_path = os.path.join(output_dir, f"prompt_pca_layer{layer_idx}.png")
    plt.savefig(pca_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved PCA: {pca_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract prompt-side Arabic dialect vectors from MADAR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--dialects", nargs="+", default=None,
                        help="City names to extract (e.g. Cairo Rabat)")
    parser.add_argument("--all-dialects", action="store_true",
                        help="Extract all available dialects")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Sentences per city (default: all)")
    parser.add_argument("--layer", type=int, default=16,
                        help="Layer index for visualization (default: 16)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing MADAR.corpus.*.tsv files")
    parser.add_argument("--no-save-vectors", action="store_true",
                        help="Do not save extracted vectors")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Re-extract even if vectors already exist")

    args = parser.parse_args()

    available_cities = discover_cities(args.data_dir)
    if not available_cities:
        print(f"Error: no MADAR.corpus.*.tsv files found in '{args.data_dir}'")
        return

    if args.all_dialects:
        dialects = available_cities
    elif args.dialects:
        dialects = args.dialects
        for d in dialects:
            if d not in available_cities:
                print(f"Error: '{d}' not found. Available: {available_cities}")
                return
    else:
        print("Error: specify --dialects or --all-dialects")
        parser.print_help()
        return

    model_name = args.model.split("/")[-1]
    output_dir = args.output_dir or f"analysis_results_dialect/{model_name}_prompt"
    save_dir   = f"dialect_vectors/{model_name}"

    print(f"\n{'='*60}")
    print("ARABIC DIALECT VECTOR EXTRACTION  (prompt-side)")
    print(f"{'='*60}")
    print(f"Model:    {args.model}")
    print(f"Cities:   {', '.join(dialects)}")
    print(f"Samples:  {args.n_samples if args.n_samples is not None else 'all'}")
    print(f"{'='*60}")

    # Load model
    print("\nLoading model...")
    load_kwargs = {"torch_dtype": torch.float16}
    try:
        import accelerate  # noqa: F401
        load_kwargs["device_map"] = "auto"
    except ImportError:
        pass

    model     = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on: {next(model.parameters()).device}")

    # Build template parts once
    prefix_str, suffix_str = build_template_parts(tokenizer)
    print(f"\nTemplate prefix: {repr(prefix_str)}")
    print(f"Template suffix: {repr(suffix_str)}")

    # Extract
    all_vectors = {}
    for dialect in dialects:
        vector_path = f"{save_dir}/{dialect}_prompt_avg_diff.pt"
        try:
            if os.path.exists(vector_path) and not args.force_reextract:
                print(f"\n  Loading existing vector: {vector_path}")
                vec = torch.load(vector_path, weights_only=False)
                all_vectors[dialect] = vec
                print(f"  Shape: {vec.shape}")
            else:
                vec = extract_dialect_vector(
                    model, tokenizer, dialect,
                    prefix_str, suffix_str,
                    n_samples=args.n_samples,
                    data_dir=args.data_dir,
                )
                all_vectors[dialect] = vec
                if not args.no_save_vectors:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(vec, vector_path)
                    print(f"  Saved: {vector_path}  shape={vec.shape}")
        except Exception as e:
            print(f"  Error processing '{dialect}': {e}")
            continue

    if not all_vectors:
        print("No vectors extracted.")
        return

    visualize_dialect_vectors(all_vectors, output_dir, layer_idx=args.layer)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"\nVectors saved in: {save_dir}/  (*_prompt_avg_diff.pt)")
    print(f"\nUse for prompt steering:")
    first = list(all_vectors.keys())[0]
    print(f"  python generate_steered_responses.py \\")
    print(f"      --model {args.model} \\")
    print(f"      --vector-path {save_dir}/{first}_prompt_avg_diff.pt \\")
    print(f"      --steering-type prompt \\")
    print(f"      --layers <layer> --coef <coef>")


if __name__ == "__main__":
    main()
