"""
Extract Arabic dialect vectors from MADAR corpus TSV files and visualize with PCA.

For each target city/dialect, parallel MADAR sentences are used directly as
responses. Positive = city dialect text, Negative = the MSA translation of the
same sentence (matched by sentID). Hidden states are extracted from the text
portion and the difference (dialect − MSA) is computed per layer.

TSV files expected:
    MADAR.corpus.{City}.tsv   — dialect side
    MADAR.corpus.MSA.tsv      — MSA side (shared negative)

Usage:
    python extract_dialect_vectors_fast.py --model Qwen/Qwen2.5-7B-Instruct --dialects Cairo Alexandria
    python extract_dialect_vectors_fast.py --model Qwen/Qwen2.5-7B-Instruct --all-dialects
    python extract_dialect_vectors_fast.py --model Qwen/Qwen2.5-7B-Instruct --dialects Cairo --layer 20 --n-samples 50
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

# Generic Arabic prompt used as the conversation turn preceding each MADAR sample
EXTRACTION_PROMPT = "أجب بجملة واحدة."


# ─── Data Loading ─────────────────────────────────────────────────────────────

def discover_cities(data_dir="."):
    """Return sorted list of city names from MADAR.corpus.{City}.tsv files (excludes MSA)."""
    tsv_files = Path(data_dir).glob("MADAR.corpus.*.tsv")
    cities = []
    for f in sorted(tsv_files):
        city = f.stem.replace("MADAR.corpus.", "")
        if city != "MSA":
            cities.append(city)
    return cities


def load_tsv(path):
    """Load a MADAR corpus TSV and return a DataFrame indexed by sentID."""
    df = pd.read_csv(path, sep="\t", dtype={"sentID.BTEC": int})
    df = df.rename(columns={"sentID.BTEC": "sentID", "sent": "text"})
    df = df.set_index("sentID")
    return df


def load_parallel_samples(city, n=None, data_dir="."):
    """
    Load parallel sentence pairs (city dialect vs MSA), matched and sorted by sentID.
    If n is None, all common parallel sentences are used.

    Returns:
        pos_texts: list of city dialect sentences
        neg_texts: list of MSA sentences (same sentIDs, same order)
        sent_ids:  list of sentence IDs used
    """
    city_path = Path(data_dir) / f"MADAR.corpus.{city}.tsv"
    msa_path  = Path(data_dir) / MSA_TSV_FILENAME

    if not city_path.exists():
        raise FileNotFoundError(f"City TSV not found: {city_path}")
    if not msa_path.exists():
        raise FileNotFoundError(f"MSA TSV not found: {msa_path}")

    city_df = load_tsv(city_path)
    msa_df  = load_tsv(msa_path)

    # Keep only sentence IDs present in both files, sorted
    common_ids = sorted(city_df.index.intersection(msa_df.index))

    if n is None:
        n = len(common_ids)
    elif n > len(common_ids):
        print(f"  Warning: only {len(common_ids)} parallel sentences for '{city}', requested {n}.")
        n = len(common_ids)

    selected_ids = common_ids[:n]

    pos_texts = city_df.loc[selected_ids, "text"].tolist()
    neg_texts = msa_df.loc[selected_ids, "text"].tolist()

    return pos_texts, neg_texts, selected_ids


# ─── Hidden State Extraction ──────────────────────────────────────────────────

def get_response_hidden_states(model, tokenizer, prompts, responses, layer_list=None):
    """
    Extract response hidden states (the MADAR text portion only).

    Args:
        prompts:   List of formatted prompt strings (with chat template applied).
        responses: List of dialect text strings.
        layer_list: Which layers to extract (default: all).

    Returns:
        response_avg: List of tensors (one per layer) containing averaged activations.
    """
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer + 1))

    response_avg = [[] for _ in range(max_layer + 1)]
    texts = [p + r for p, r in zip(prompts, responses)]

    for text, prompt in tqdm(zip(texts, prompts), total=len(texts), desc="Extracting hidden states"):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer in layer_list:
            response_avg[layer].append(
                outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()
            )

        del outputs

    for layer in layer_list:
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)

    return response_avg


def build_prompts(tokenizer, texts):
    """
    Wrap each MADAR text in a minimal chat template so that we have a
    clean prompt/response split.  The text itself is the assistant turn.

    Returns:
        prompts:   List of formatted prompt strings (everything before the text).
        responses: The original texts (unchanged).
    """
    base_messages = [{"role": "user", "content": EXTRACTION_PROMPT}]
    base_prompt = tokenizer.apply_chat_template(
        base_messages, tokenize=False, add_generation_prompt=True
    )
    prompts = [base_prompt] * len(texts)
    return prompts, texts


# ─── Vector Extraction ────────────────────────────────────────────────────────

def extract_dialect_vector(model, tokenizer, city, n_samples=None, data_dir="."):
    """
    Extract a dialect direction vector (city dialect − MSA) using parallel sentences.

    Returns:
        response_avg_diff: Tensor of shape [num_layers, hidden_dim]
    """
    print(f"\n{'='*60}")
    print(f"Extracting vector for city: {city}")
    print(f"{'='*60}")

    pos_texts, neg_texts, sent_ids = load_parallel_samples(city, n_samples, data_dir)

    print(f"  Parallel sentences used: {len(sent_ids)}")
    print(f"  sentID range: {sent_ids[0]} – {sent_ids[-1]}")
    print(f"  Positive ({city}) | Negative (MSA) — first 3 pairs:")
    for i in range(min(3, len(sent_ids))):
        print(f"    [{sent_ids[i]}] + {pos_texts[i]}")
        print(f"    [{sent_ids[i]}] - {neg_texts[i]}")
        print()

    # Save selected samples to a TSV for inspection
    samples_path = os.path.join(data_dir, f"selected_samples_{city}.tsv")
    samples_df = pd.DataFrame({
        "sentID": sent_ids,
        f"positive_{city}": pos_texts,
        "negative_MSA": neg_texts,
    })
    samples_df.to_csv(samples_path, sep="\t", index=False)
    print(f"  Saved selected samples: {samples_path}")

    pos_prompts, pos_responses = build_prompts(tokenizer, pos_texts)
    neg_prompts, neg_responses = build_prompts(tokenizer, neg_texts)

    print(f"  Extracting positive ({city}) hidden states...")
    pos_avg = get_response_hidden_states(model, tokenizer, pos_prompts, pos_responses)

    print(f"  Extracting negative (MSA) hidden states...")
    neg_avg = get_response_hidden_states(model, tokenizer, neg_prompts, neg_responses)

    print(f"\n  Computing diff across {len(pos_avg)} layers...")
    max_layers = len(pos_avg)
    response_avg_diff = torch.stack([
        pos_avg[l].mean(0).float() - neg_avg[l].mean(0).float()
        for l in range(max_layers)
    ], dim=0)

    return response_avg_diff


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_dialect_vectors(vectors_dict, output_dir, layer_idx=16):
    """
    Visualize dialect vectors with a cosine-similarity heatmap and PCA plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    dialects = list(vectors_dict.keys())
    print(f"\n{'='*60}")
    print(f"Visualizing {len(dialects)} dialects: {', '.join(dialects)}")
    print(f"Using layer {layer_idx}")
    print(f"{'='*60}")

    layer_vectors = {}
    for dialect, vector in vectors_dict.items():
        if vector.shape[0] > layer_idx:
            layer_vectors[dialect] = vector[layer_idx]
        else:
            print(f"  Warning: '{dialect}' has fewer than {layer_idx+1} layers, skipping.")

    if len(layer_vectors) < 2:
        print("  Error: Need at least 2 vectors for analysis.")
        return

    dialects = list(layer_vectors.keys())

    # --- Cosine Similarity Heatmap ---
    print("\n  Creating cosine similarity heatmap...")
    normalized = {d: v / torch.linalg.norm(v) for d, v in layer_vectors.items()}
    sim_matrix = pd.DataFrame(index=dialects, columns=dialects, dtype=float)
    for d1 in dialects:
        for d2 in dialects:
            sim_matrix.loc[d1, d2] = torch.dot(normalized[d1], normalized[d2]).item()

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, cmap="viridis", fmt=".2f",
                cbar_kws={"label": "Cosine Similarity"})
    plt.title(f"Arabic Dialect Vector Cosine Similarity (Layer {layer_idx})",
              fontsize=16, fontweight="bold")
    plt.xlabel("Dialect", fontsize=12)
    plt.ylabel("Dialect", fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"dialect_similarity_heatmap_layer{layer_idx}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {heatmap_path}")

    # --- PCA 2D ---
    print("\n  Creating 2D PCA visualization...")
    vector_matrix = torch.stack([layer_vectors[d] for d in dialects]).numpy()
    n_components = min(5, len(dialects))
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(vector_matrix)

    print(f"  Explained variance: {pca.explained_variance_ratio_[:3]}")

    pca_df = pd.DataFrame(pcs[:, :2], index=dialects, columns=["PC1", "PC2"])
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=300,
                    color="steelblue", edgecolor="black", linewidth=1.5)
    plt.title(f"Arabic Dialect Vectors - PCA Projection (Layer {layer_idx})",
              fontsize=16, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=12)
    plt.grid(True, alpha=0.3)
    for i, d in enumerate(dialects):
        plt.annotate(d,
                     (pca_df["PC1"].iloc[i], pca_df["PC2"].iloc[i]),
                     xytext=(5, 5), textcoords="offset points",
                     fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    plt.tight_layout()
    pca_path = os.path.join(output_dir, f"pca_dialect_vectors_layer{layer_idx}.png")
    plt.savefig(pca_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved 2D PCA: {pca_path}")

    # --- PCA 3D (color-encoded) ---
    if len(dialects) >= 3 and n_components >= 3:
        print("\n  Creating 3D PCA visualization...")
        pca_df_3d = pd.DataFrame(pcs[:, :3], index=dialects, columns=["PC1", "PC2", "PC3"])
        plt.figure(figsize=(14, 9))
        scatter = sns.scatterplot(
            data=pca_df_3d, x="PC1", y="PC2", hue="PC3",
            palette="coolwarm", s=300, edgecolor="black", linewidth=1.5, legend="full"
        )
        plt.title(f"Arabic Dialect Vectors - 3D PCA (Layer {layer_idx})",
                  fontsize=16, fontweight="bold")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=12)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=12)
        plt.grid(True, alpha=0.3)
        for i, d in enumerate(dialects):
            plt.annotate(d,
                         (pca_df_3d["PC1"].iloc[i], pca_df_3d["PC2"].iloc[i]),
                         xytext=(5, 5), textcoords="offset points",
                         fontsize=10, fontweight="bold")
        scatter.legend(title=f"PC3 ({pca.explained_variance_ratio_[2]:.2%})",
                       loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        pca_3d_path = os.path.join(output_dir, f"pca_3d_dialect_vectors_layer{layer_idx}.png")
        plt.savefig(pca_3d_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved 3D PCA: {pca_3d_path}")

    # --- Interpretation ---
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    loadings_df = pd.DataFrame(pcs, index=dialects,
                               columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
    print("\n  PC1 (strongest axis):")
    print("    Most positive:", loadings_df["PC1"].nlargest(3).to_dict())
    print("    Most negative:", loadings_df["PC1"].nsmallest(3).to_dict())
    if pcs.shape[1] >= 2:
        print("\n  PC2:")
        print("    Most positive:", loadings_df["PC2"].nlargest(3).to_dict())
        print("    Most negative:", loadings_df["PC2"].nsmallest(3).to_dict())

    print(f"\n  All visualizations saved to: {output_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract Arabic dialect vectors from MADAR data and visualize with PCA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_dialect_vectors_fast.py --model Qwen/Qwen2.5-7B-Instruct --dialects Cairo Alexandria
  python extract_dialect_vectors_fast.py --model Qwen/Qwen2.5-7B-Instruct --all-dialects
  python extract_dialect_vectors_fast.py --model Qwen/Qwen2.5-7B-Instruct --dialects Cairo --layer 20 --n-samples 50
  python extract_dialect_vectors_fast.py --model Qwen/Qwen2.5-7B-Instruct --all-dialects --data-dir /path/to/tsv/files
        """
    )

    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--dialects", nargs="+", default=None,
                        help="Dialects to extract (e.g. Levantine Egyptian Gulf Moroccan)")
    parser.add_argument("--all-dialects", action="store_true",
                        help="Extract all dialects (Levantine, Egyptian, Gulf, Moroccan)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Samples per city (default: all common parallel sentences)")
    parser.add_argument("--layer", type=int, default=16,
                        help="Layer index to visualize (default: 16)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for visualizations")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing MADAR.corpus.*.tsv files (default: .)")
    parser.add_argument("--save-vectors", action="store_true", default=True,
                        help="Save extracted vectors (default: True)")
    parser.add_argument("--no-save-vectors", action="store_false", dest="save_vectors",
                        help="Don't save extracted vectors")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Re-extract even if vectors already exist")

    args = parser.parse_args()

    # Determine cities
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
                print(f"Error: '{d}' not found. Available cities: {available_cities}")
                return
    else:
        print("Error: specify --dialects or --all-dialects")
        parser.print_help()
        return

    model_name = args.model.split("/")[-1]
    output_dir = args.output_dir or f"analysis_results_dialect/{model_name}"
    save_dir = f"dialect_vectors/{model_name}"

    print(f"\n{'='*60}")
    print("ARABIC DIALECT VECTOR EXTRACTION")
    print(f"{'='*60}")
    print(f"Model:               {args.model}")
    print(f"Cities:              {', '.join(dialects)}")
    print(f"Negative baseline:   MSA (parallel sentences)")
    print(f"Samples per city:    {args.n_samples if args.n_samples is not None else 'all common parallel'}")
    print(f"Visualization layer: {args.layer}")
    print(f"Data directory:      {args.data_dir}")
    print(f"Output directory:    {output_dir}")
    print(f"{'='*60}")

    # Check existing vectors
    existing, missing = [], []
    for d in dialects:
        vpath = f"{save_dir}/{d}_response_avg_diff.pt"
        (existing if os.path.exists(vpath) else missing).append(d)

    if existing and not args.force_reextract:
        print(f"\n  Found existing vectors: {', '.join(existing)}")
    if missing:
        print(f"  Need to extract: {', '.join(missing)}")

    # Load model
    print("\nLoading model...")
    load_kwargs = {"torch_dtype": torch.float16}
    try:
        import accelerate  # noqa: F401
        load_kwargs["device_map"] = "auto"
    except ImportError:
        print("  Note: accelerate not found, loading without device_map (will use CPU or single GPU)")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    if device.type == "cpu":
        print("  Warning: running on CPU — consider reducing --n-samples")
    else:
        print(f"  Model loaded on: {device}")

    # Extract
    all_vectors = {}
    for dialect in dialects:
        vector_path = f"{save_dir}/{dialect}_response_avg_diff.pt"
        try:
            if os.path.exists(vector_path) and not args.force_reextract:
                print(f"\n  Loading existing vector: {vector_path}")
                vec = torch.load(vector_path, weights_only=False)
                all_vectors[dialect] = vec
                print(f"  Shape: {vec.shape}")
            else:
                vec = extract_dialect_vector(
                    model, tokenizer, dialect,
                    n_samples=args.n_samples,
                    data_dir=args.data_dir,
                )
                all_vectors[dialect] = vec
                if args.save_vectors:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(vec, vector_path)
                    print(f"  Saved: {vector_path}  shape={vec.shape}")
        except Exception as e:
            print(f"  Error processing '{dialect}': {e}")
            continue

    if not all_vectors:
        print("No vectors extracted. Exiting.")
        return

    # Visualize
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    visualize_dialect_vectors(all_vectors, output_dir, layer_idx=args.layer)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    if args.save_vectors and all_vectors:
        print(f"\n  Saved vectors in: dialect_vectors/{model_name}/")
        print(f"  Cities: {', '.join(all_vectors.keys())}")
        print(f"\n  Use for steering:")
        first = list(all_vectors.keys())[0]
        print(f"    python steer_dialect_and_compare.py \\")
        print(f"        --model {args.model} \\")
        print(f"        --vector-path dialect_vectors/{model_name}/{first}_response_avg_diff.pt \\")
        print(f"        --steer-dialect {first} \\")
        print(f"        --layer {args.layer} --coef 3.0")


if __name__ == "__main__":
    main()
