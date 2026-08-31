"""
Visualize cosine similarity between dialect steering vectors across models.

Loads pre-extracted .pt vector files and produces:
  1. Cosine similarity heatmap at a chosen layer
  2. PCA 2D + 3D scatter at a chosen layer
  3. Mean pairwise cosine similarity across all layers (line plot)
  4. Per-pair cosine similarity across all layers (heatmap: pairs × layers)
  5. CSV report of mean similarity per layer

No model loading required — works purely from saved .pt files.

Usage:
    python visualize_vector_similarity.py --layer 16
    python visualize_vector_similarity.py --layer 16 --output-dir my_plots
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pathlib import Path
from sklearn.decomposition import PCA


MODELS = {
    "ALLaM":  "dialect_vectors/ALLaM-7B-Instruct-preview",
    "Fanar":  "dialect_vectors/Fanar-1-9B-Instruct",
    "Jais":   "dialect_vectors/Jais-2-8B-Chat",
}

CITIES = [
    "Cairo", "Rabat", "Aleppo", "Beirut", "Damascus",
    "Doha", "Jeddah", "Khartoum", "Riyadh", "Tunis",
]

VECTOR_TYPES = {
    "response": "_response_avg_diff.pt",
    "prompt":   "_prompt_avg_diff.pt",
}


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_all_layers(model_dir, cities, suffix):
    """Load full [num_layers, hidden_dim] tensors for each available city."""
    all_vecs = {}
    for city in cities:
        path = Path(model_dir) / f"{city}{suffix}"
        if not path.exists():
            continue
        all_vecs[city] = torch.load(path, weights_only=False).float()
    return all_vecs


def slice_layer(all_vecs, layer_idx):
    """Slice a single layer from pre-loaded full tensors."""
    return {
        city: vec[layer_idx]
        for city, vec in all_vecs.items()
        if layer_idx < vec.shape[0]
    }


# ─── Similarity ───────────────────────────────────────────────────────────────

def cosine_similarity_matrix(vectors):
    """Pairwise cosine similarity. Returns DataFrame."""
    cities = list(vectors.keys())
    norm = {c: v / torch.linalg.norm(v) for c, v in vectors.items()}
    mat = pd.DataFrame(index=cities, columns=cities, dtype=float)
    for c1 in cities:
        for c2 in cities:
            mat.loc[c1, c2] = torch.dot(norm[c1], norm[c2]).item()
    return mat


def compute_cross_layer_similarities(all_vecs):
    """
    For each layer, compute all pairwise cosine similarities.

    Returns:
        layers:       list of layer indices
        mean_per_layer: list of mean off-diagonal cosine similarity per layer
        pair_df:      DataFrame [layer × pair_label] of cosine similarities
    """
    cities = list(all_vecs.keys())
    if len(cities) < 2:
        return [], [], pd.DataFrame()

    num_layers = min(v.shape[0] for v in all_vecs.values())
    pairs = list(combinations(cities, 2))
    pair_labels = [f"{a}–{b}" for a, b in pairs]

    rows = []
    means = []
    for layer in range(num_layers):
        vecs = slice_layer(all_vecs, layer)
        norm = {c: v / torch.linalg.norm(v) for c, v in vecs.items()}
        sims = [torch.dot(norm[a], norm[b]).item() for a, b in pairs]
        rows.append(sims)
        means.append(np.mean(sims))

    pair_df = pd.DataFrame(rows, columns=pair_labels)
    pair_df.index.name = "layer"
    return list(range(num_layers)), means, pair_df


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_heatmap(sim_matrix, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        sim_matrix.astype(float),
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Dialect", fontsize=11)
    ax.set_ylabel("Dialect", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pca(vectors, layer_idx, model_name, vec_type, output_dir):
    cities = list(vectors.keys())
    matrix = torch.stack([vectors[c] for c in cities]).numpy()
    n_components = min(5, len(cities))
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(matrix)

    print(f"  Explained variance (PC1-3): {pca.explained_variance_ratio_[:3]}")

    # 2D
    pca_df = pd.DataFrame(pcs[:, :2], index=cities, columns=["PC1", "PC2"])
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=300,
                    color="steelblue", edgecolor="black", linewidth=1.5)
    plt.title(f"{model_name} — {vec_type.capitalize()}-side\nPCA Projection (Layer {layer_idx})",
              fontsize=13, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=11)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=11)
    plt.grid(True, alpha=0.3)
    for i, city in enumerate(cities):
        plt.annotate(city, (pca_df["PC1"].iloc[i], pca_df["PC2"].iloc[i]),
                     xytext=(5, 5), textcoords="offset points",
                     fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    plt.tight_layout()
    path2d = os.path.join(output_dir, f"{model_name.lower()}_{vec_type}_layer{layer_idx}_pca2d.png")
    plt.savefig(path2d, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2d}")

    # 3D
    if n_components >= 3:
        pca_df_3d = pd.DataFrame(pcs[:, :3], index=cities, columns=["PC1", "PC2", "PC3"])
        plt.figure(figsize=(14, 9))
        scatter = sns.scatterplot(
            data=pca_df_3d, x="PC1", y="PC2", hue="PC3",
            palette="coolwarm", s=300, edgecolor="black", linewidth=1.5, legend="full"
        )
        plt.title(f"{model_name} — {vec_type.capitalize()}-side\nPCA 3D (Layer {layer_idx})",
                  fontsize=13, fontweight="bold")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=11)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=11)
        plt.grid(True, alpha=0.3)
        for i, city in enumerate(cities):
            plt.annotate(city, (pca_df_3d["PC1"].iloc[i], pca_df_3d["PC2"].iloc[i]),
                         xytext=(5, 5), textcoords="offset points",
                         fontsize=10, fontweight="bold")
        scatter.legend(title=f"PC3 ({pca.explained_variance_ratio_[2]:.2%})",
                       loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        path3d = os.path.join(output_dir, f"{model_name.lower()}_{vec_type}_layer{layer_idx}_pca3d.png")
        plt.savefig(path3d, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path3d}")


def plot_mean_similarity_across_layers(layers, means, model_name, vec_type, output_dir):
    """Line plot: mean pairwise cosine similarity vs layer."""
    plt.figure(figsize=(10, 5))
    plt.plot(layers, means, marker="o", markersize=4, linewidth=1.8, color="steelblue")
    plt.title(f"{model_name} — {vec_type.capitalize()}-side\n"
              f"Mean Pairwise Cosine Similarity Across Layers",
              fontsize=13, fontweight="bold")
    plt.xlabel("Layer", fontsize=11)
    plt.ylabel("Mean Cosine Similarity", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{model_name.lower()}_{vec_type}_mean_sim_by_layer.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pair_similarity_across_layers(pair_df, model_name, vec_type, output_dir):
    """Heatmap: dialect pairs (rows) × layers (columns), color = cosine similarity."""
    fig, ax = plt.subplots(figsize=(max(14, len(pair_df) // 2), 10))
    sns.heatmap(
        pair_df.T.astype(float),
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        ax=ax, cbar_kws={"label": "Cosine Similarity", "shrink": 0.6},
        xticklabels=max(1, len(pair_df) // 20),
    )
    ax.set_title(f"{model_name} — {vec_type.capitalize()}-side\n"
                 f"Per-Pair Cosine Similarity Across Layers",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Dialect Pair", fontsize=9)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{model_name.lower()}_{vec_type}_pair_sim_by_layer.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity and PCA for dialect steering vectors"
    )
    parser.add_argument("--layer", type=int, default=16,
                        help="Layer index for single-layer plots (default: 16)")
    parser.add_argument("--output-dir", default="analysis_results_dialect/similarity",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for vec_type, suffix in VECTOR_TYPES.items():
        print(f"\n{'='*60}")
        print(f"Vector type: {vec_type}")
        print(f"{'='*60}")

        for model_name, model_dir in MODELS.items():
            print(f"\n  Model: {model_name}")

            # Load all layers once
            all_vecs = load_all_layers(model_dir, CITIES, suffix)
            if len(all_vecs) < 2:
                print(f"  Not enough vectors found — skipping.")
                continue

            print(f"  Cities loaded: {list(all_vecs.keys())}")

            # ── Single-layer plots ────────────────────────────────────────────
            vectors_at_layer = slice_layer(all_vecs, args.layer)
            if len(vectors_at_layer) >= 2:
                sim_matrix = cosine_similarity_matrix(vectors_at_layer)
                plot_heatmap(
                    sim_matrix,
                    title=(f"{model_name} — {vec_type.capitalize()}-side Dialect Vectors\n"
                           f"Cosine Similarity (Layer {args.layer})"),
                    save_path=os.path.join(args.output_dir,
                        f"{model_name.lower()}_{vec_type}_layer{args.layer}_cosine_sim.png"),
                )
                plot_pca(vectors_at_layer, args.layer, model_name, vec_type, args.output_dir)

            # ── Cross-layer plots ─────────────────────────────────────────────
            print(f"\n  Computing cross-layer similarities...")
            layers, means, pair_df = compute_cross_layer_similarities(all_vecs)

            if layers:
                plot_mean_similarity_across_layers(layers, means, model_name, vec_type, args.output_dir)
                plot_pair_similarity_across_layers(pair_df, model_name, vec_type, args.output_dir)

                # Save CSV report
                csv_path = os.path.join(args.output_dir,
                    f"{model_name.lower()}_{vec_type}_mean_sim_by_layer.csv")
                pd.DataFrame({"layer": layers, "mean_cosine_similarity": means}).to_csv(
                    csv_path, index=False)
                print(f"  Saved: {csv_path}")

                pair_csv = os.path.join(args.output_dir,
                    f"{model_name.lower()}_{vec_type}_pair_sim_by_layer.csv")
                pair_df.to_csv(pair_csv)
                print(f"  Saved: {pair_csv}")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
