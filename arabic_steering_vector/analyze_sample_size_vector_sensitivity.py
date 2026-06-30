#!/usr/bin/env python3
"""Analyze steering-vector sensitivity to construction sample size.

The expected input layout matches run_sampled_cairo_msa_vectors.sh:

    sample_split_vector_runs/
      1k/dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt
      2k/dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt
      ...

The script writes cosine-similarity CSVs plus PCA/diagnostic plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_SIZES = ("1k", "2k", "4k", "6k", "12k")
VECTOR_KEYS = ("response_avg_diff", "vector", "steering_vector", "dialect_vector", "vectors")


def parse_size_value(label: str) -> int:
    text = label.strip().lower()
    if not text:
        raise argparse.ArgumentTypeError("Size labels cannot be empty.")
    if text.endswith("k"):
        number = text[:-1]
        if not number.isdigit():
            raise argparse.ArgumentTypeError(f"Invalid size label: {label}")
        return int(number) * 1000
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"Invalid size label: {label}")
    return int(text)


def sorted_size_labels(labels: list[str]) -> list[str]:
    return sorted(labels, key=parse_size_value)


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "This script needs PyTorch to load .pt steering vectors. "
            "Run it in the same environment you used for vector extraction."
        ) from exc
    return torch


def maybe_import_matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, labels: list[str], matrix: list[list[float]]) -> None:
    rows = []
    for label, values in zip(labels, matrix):
        row: dict[str, Any] = {"size": label}
        row.update({col: value for col, value in zip(labels, values)})
        rows.append(row)
    write_rows(path, ["size", *labels], rows)


def mean_or_nan(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def min_or_nan(values: list[float]) -> float:
    return min(values) if values else float("nan")


def max_or_nan(values: list[float]) -> float:
    return max(values) if values else float("nan")


def vector_path_for(
    runs_dir: Path,
    size: str,
    model_name: str,
    dialect: str,
    vector_filename: str,
    path_template: str | None,
) -> Path:
    if path_template:
        rendered = path_template.format(
            runs_dir=str(runs_dir),
            size=size,
            model_name=model_name,
            dialect=dialect,
            vector_filename=vector_filename,
        )
        return Path(rendered)
    return runs_dir / size / "dialect_vectors" / model_name / vector_filename


def discover_size_labels(
    runs_dir: Path,
    model_name: str,
    dialect: str,
    vector_filename: str,
    path_template: str | None,
) -> list[str]:
    if not runs_dir.exists():
        return []

    labels = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            parse_size_value(child.name)
        except argparse.ArgumentTypeError:
            continue

        path = vector_path_for(
            runs_dir=runs_dir,
            size=child.name,
            model_name=model_name,
            dialect=dialect,
            vector_filename=vector_filename,
            path_template=path_template,
        )
        if path.exists():
            labels.append(child.name)

    return sorted_size_labels(labels)


def tensor_from_loaded_object(obj: Any, torch: Any, path: Path) -> Any:
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        for key in VECTOR_KEYS:
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key]

        tensor_values = [value for value in obj.values() if torch.is_tensor(value)]
        if len(tensor_values) == 1:
            return tensor_values[0]

    raise ValueError(
        f"Could not find a tensor in {path}. Expected a tensor or a dict with one of: "
        f"{', '.join(VECTOR_KEYS)}"
    )


def load_vector(path: Path, torch: Any) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)

    tensor = tensor_from_loaded_object(obj, torch, path).detach().float().cpu()

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"{path} has shape {tuple(tensor.shape)}. Expected [num_layers, hidden_dim]."
        )
    return tensor


def safe_cosine(a: Any, b: Any, torch: Any) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() == 0:
        return float("nan")
    return (torch.dot(a, b) / denom).item()


def vector_norm(vector: Any, torch: Any) -> float:
    return torch.linalg.vector_norm(vector).item()


def resolve_layer(requested_layer: int, num_layers: int) -> int:
    layer = requested_layer
    if layer < 0:
        layer = num_layers + layer
    if layer < 0 or layer >= num_layers:
        raise ValueError(
            f"Layer {requested_layer} resolves to {layer}, but vectors have layers 0..{num_layers - 1}."
        )
    return layer


def normalize_rows(matrix: Any, torch: Any, eps: float = 1e-12) -> Any:
    norms = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
    return matrix / norms.clamp_min(eps)


def run_pca(matrix: Any, torch: Any, normalize: bool) -> tuple[Any, list[float]]:
    if matrix.shape[0] < 2:
        raise ValueError("PCA needs at least two vectors.")

    x = matrix.float().cpu()
    if normalize:
        x = normalize_rows(x, torch)

    centered = x - x.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)

    n_components = min(2, vh.shape[0])
    coords = centered @ vh[:n_components].T
    if n_components == 1:
        coords = torch.cat([coords, torch.zeros_like(coords)], dim=1)

    variances = singular_values.pow(2)
    total = variances.sum().item()
    if total > 0:
        explained = (variances / total).tolist()
    else:
        explained = [0.0 for _ in range(len(variances))]

    explained = (explained + [0.0, 0.0])[:2]
    return coords[:, :2], explained


def plot_pairwise_heatmap(
    plt: Any,
    path: Path,
    labels: list[str],
    matrix: list[list[float]],
    title: str,
    dpi: int,
) -> None:
    fig_size = max(5.0, 0.75 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if math.isnan(value):
                text = "nan"
            else:
                text = f"{value:.3f}"
            color = "white" if not math.isnan(value) and abs(value) > 0.65 else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_lines_by_layer(
    plt: Any,
    path: Path,
    rows: list[dict[str, Any]],
    group_key: str,
    value_key: str,
    title: str,
    ylabel: str,
    dpi: int,
) -> None:
    groups = sorted({str(row[group_key]) for row in rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    for group in groups:
        group_rows = [row for row in rows if str(row[group_key]) == group]
        group_rows.sort(key=lambda row: int(row["layer"]))
        ax.plot(
            [int(row["layer"]) for row in group_rows],
            [float(row[value_key]) for row in group_rows],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=group,
        )
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_size_series(
    plt: Any,
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    dpi: int,
) -> None:
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, values, marker="o", linewidth=2)
    ax.set_xticks(x, labels=labels)
    ax.set_title(title)
    ax.set_xlabel("Sample size")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_pca_layer(
    plt: Any,
    path: Path,
    labels: list[str],
    coords: Any,
    explained: list[float],
    title: str,
    dpi: int,
) -> None:
    xs = [coords[i, 0].item() for i in range(coords.shape[0])]
    ys = [coords[i, 1].item() for i in range(coords.shape[0])]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(xs, ys, color="0.55", linewidth=1.5, alpha=0.8)
    ax.scatter(xs, ys, s=70, zorder=3)
    for label, x, y in zip(labels, xs, ys):
        ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)")
    ax.axhline(0, color="0.8", linewidth=0.8)
    ax.axvline(0, color="0.8", linewidth=0.8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_pca_all_layers(
    plt: Any,
    path: Path,
    pca_rows: list[dict[str, Any]],
    labels: list[str],
    selected_layer: int,
    explained: list[float],
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in labels:
        group = [row for row in pca_rows if row["size"] == label]
        group.sort(key=lambda row: int(row["layer"]))
        ax.scatter(
            [float(row["pc1"]) for row in group],
            [float(row["pc2"]) for row in group],
            s=22,
            alpha=0.65,
            label=label,
        )
        selected = [row for row in group if int(row["layer"]) == selected_layer]
        if selected:
            ax.scatter(
                [float(row["pc1"]) for row in selected],
                [float(row["pc2"]) for row in selected],
                s=90,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
            )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)")
    ax.axhline(0, color="0.8", linewidth=0.8)
    ax.axvline(0, color="0.8", linewidth=0.8)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    labels: list[str],
    layer: int,
    reference_size: str,
    vector_paths: dict[str, Path],
    summary_rows: list[dict[str, Any]],
    adjacent_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "Sample-size steering vector sensitivity analysis",
        "",
        f"Runs directory: {args.runs_dir}",
        f"Model directory: {args.model_name}",
        f"Dialect: {args.dialect}",
        f"Vector filename: {args.vector_filename}",
        f"Sizes: {', '.join(labels)}",
        f"Reference size: {reference_size}",
        f"Focused layer: {layer}",
        f"PCA uses normalized vectors: {args.pca_normalize}",
        "",
        "Loaded vectors:",
    ]
    for label in labels:
        lines.append(f"- {label}: {vector_paths[label]}")

    lines.extend(["", f"Cosine to {reference_size} at layer {layer}:"])
    for row in summary_rows:
        lines.append(
            f"- {row['size']}: {float(row['cosine_to_reference_layer']):.6f} "
            f"(mean across layers: {float(row['mean_cosine_to_reference_all_layers']):.6f})"
        )

    layer_adjacent = [row for row in adjacent_rows if int(row["layer"]) == layer]
    if layer_adjacent:
        lines.extend(["", f"Adjacent-size cosine at layer {layer}:"])
        for row in layer_adjacent:
            lines.append(f"- {row['comparison']}: {float(row['cosine']):.6f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compare steering vectors built from different sample sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_sample_size_vector_sensitivity.py

  python analyze_sample_size_vector_sensitivity.py \\
      --runs-dir sample_split_vector_runs \\
      --model-name ALLaM-7B-Instruct-preview \\
      --dialect Cairo \\
      --sizes 1k 2k 4k 6k 12k \\
      --layer 16
        """,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=script_dir / "sample_split_vector_runs",
        help="Directory containing one subdirectory per sample size.",
    )
    parser.add_argument(
        "--model-name",
        default="ALLaM-7B-Instruct-preview",
        help="Model directory name under dialect_vectors/.",
    )
    parser.add_argument("--dialect", default="Cairo", help="Dialect/vector prefix.")
    parser.add_argument(
        "--vector-filename",
        default=None,
        help="Vector filename. Defaults to '<dialect>_response_avg_diff.pt'.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        help=(
            "Sample-size labels to compare, e.g. 1k 2k 4k 6k 12k. "
            "If omitted, the script discovers available vector directories."
        ),
    )
    parser.add_argument(
        "--reference-size",
        default=None,
        help="Reference size for convergence comparisons. Defaults to the largest --sizes value.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Focused layer for matrix/PCA plots. Negative values count from the end.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory. Defaults to sample_size_sensitivity_analysis/<model>/<dialect>.",
    )
    parser.add_argument(
        "--path-template",
        default=None,
        help=(
            "Optional vector path template with placeholders: {runs_dir}, {size}, "
            "{model_name}, {dialect}, {vector_filename}."
        ),
    )
    parser.add_argument(
        "--no-pca-normalize",
        action="store_false",
        dest="pca_normalize",
        help="Run PCA on raw vectors instead of row-normalized direction vectors.",
    )
    parser.set_defaults(pca_normalize=True)
    parser.add_argument(
        "--plot-each-layer-pca",
        action="store_true",
        help="Also write one PCA trajectory plot per layer.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV/text outputs.")
    parser.add_argument("--dpi", type=int, default=180, help="Plot DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.vector_filename = args.vector_filename or f"{args.dialect}_response_avg_diff.pt"
    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent
            / "sample_size_sensitivity_analysis"
            / args.model_name
            / args.dialect
        )

    if args.sizes:
        labels = sorted_size_labels(args.sizes)
    else:
        labels = discover_size_labels(
            runs_dir=args.runs_dir,
            model_name=args.model_name,
            dialect=args.dialect,
            vector_filename=args.vector_filename,
            path_template=args.path_template,
        )
        if not labels:
            labels = sorted_size_labels(list(DEFAULT_SIZES))

    if len(labels) < 2:
        raise SystemExit("At least two sample sizes are required for sensitivity analysis.")

    reference_size = args.reference_size or labels[-1]
    if reference_size not in labels:
        raise SystemExit(f"--reference-size '{reference_size}' is not in sizes: {', '.join(labels)}")

    torch = require_torch()

    vector_paths: dict[str, Path] = {}
    vectors: dict[str, Any] = {}
    for label in labels:
        path = vector_path_for(
            runs_dir=args.runs_dir,
            size=label,
            model_name=args.model_name,
            dialect=args.dialect,
            vector_filename=args.vector_filename,
            path_template=args.path_template,
        )
        vector_paths[label] = path
        try:
            vectors[label] = load_vector(path, torch)
        except FileNotFoundError as exc:
            raise SystemExit(f"Missing vector for size '{label}': {path}") from exc

    shapes = {label: tuple(vector.shape) for label, vector in vectors.items()}
    if len(set(shapes.values())) != 1:
        details = "\n".join(f"  {label}: {shape}" for label, shape in shapes.items())
        raise SystemExit(f"Vector shapes do not match:\n{details}")

    num_layers, hidden_dim = next(iter(vectors.values())).shape
    layer = resolve_layer(args.layer, num_layers)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(labels)} vectors with shape [{num_layers}, {hidden_dim}]")
    print(f"Focused layer: {layer}")
    print(f"Reference size: {reference_size}")
    print(f"Writing analysis to: {out_dir}")

    pairwise_layer = []
    for label_a in labels:
        row = []
        for label_b in labels:
            row.append(safe_cosine(vectors[label_a][layer], vectors[label_b][layer], torch))
        pairwise_layer.append(row)
    write_matrix_csv(out_dir / f"pairwise_cosine_layer{layer}.csv", labels, pairwise_layer)

    pairwise_all_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    adjacent_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []

    for current_layer in range(num_layers):
        for label_a in labels:
            norm_rows.append(
                {
                    "size": label_a,
                    "layer": current_layer,
                    "l2_norm": vector_norm(vectors[label_a][current_layer], torch),
                }
            )

            reference_rows.append(
                {
                    "size": label_a,
                    "reference_size": reference_size,
                    "layer": current_layer,
                    "cosine": safe_cosine(
                        vectors[label_a][current_layer],
                        vectors[reference_size][current_layer],
                        torch,
                    ),
                }
            )

            for label_b in labels:
                pairwise_all_rows.append(
                    {
                        "layer": current_layer,
                        "size_a": label_a,
                        "size_b": label_b,
                        "cosine": safe_cosine(
                            vectors[label_a][current_layer],
                            vectors[label_b][current_layer],
                            torch,
                        ),
                    }
                )

        for label_a, label_b in zip(labels, labels[1:]):
            adjacent_rows.append(
                {
                    "layer": current_layer,
                    "size_a": label_a,
                    "size_b": label_b,
                    "comparison": f"{label_a}_vs_{label_b}",
                    "cosine": safe_cosine(
                        vectors[label_a][current_layer],
                        vectors[label_b][current_layer],
                        torch,
                    ),
                }
            )

    write_rows(
        out_dir / "pairwise_cosine_all_layers.csv",
        ["layer", "size_a", "size_b", "cosine"],
        pairwise_all_rows,
    )
    write_rows(
        out_dir / "cosine_to_reference_by_layer.csv",
        ["size", "reference_size", "layer", "cosine"],
        reference_rows,
    )
    write_rows(
        out_dir / "adjacent_size_cosine_by_layer.csv",
        ["layer", "size_a", "size_b", "comparison", "cosine"],
        adjacent_rows,
    )
    write_rows(out_dir / "vector_norms_by_layer.csv", ["size", "layer", "l2_norm"], norm_rows)

    summary_rows: list[dict[str, Any]] = []
    for label in labels:
        ref_values = [
            float(row["cosine"])
            for row in reference_rows
            if row["size"] == label and not math.isnan(float(row["cosine"]))
        ]
        norm_values = [
            float(row["l2_norm"])
            for row in norm_rows
            if row["size"] == label and not math.isnan(float(row["l2_norm"]))
        ]
        layer_ref = [
            float(row["cosine"])
            for row in reference_rows
            if row["size"] == label and int(row["layer"]) == layer
        ][0]
        layer_norm = [
            float(row["l2_norm"])
            for row in norm_rows
            if row["size"] == label and int(row["layer"]) == layer
        ][0]
        summary_rows.append(
            {
                "size": label,
                "reference_size": reference_size,
                "cosine_to_reference_layer": layer_ref,
                "mean_cosine_to_reference_all_layers": mean_or_nan(ref_values),
                "min_cosine_to_reference_all_layers": min_or_nan(ref_values),
                "max_cosine_to_reference_all_layers": max_or_nan(ref_values),
                "l2_norm_layer": layer_norm,
                "mean_l2_norm_all_layers": mean_or_nan(norm_values),
            }
        )

    write_rows(
        out_dir / "summary.csv",
        [
            "size",
            "reference_size",
            "cosine_to_reference_layer",
            "mean_cosine_to_reference_all_layers",
            "min_cosine_to_reference_all_layers",
            "max_cosine_to_reference_all_layers",
            "l2_norm_layer",
            "mean_l2_norm_all_layers",
        ],
        summary_rows,
    )

    layer_matrix = torch.stack([vectors[label][layer] for label in labels], dim=0)
    layer_coords, layer_explained = run_pca(layer_matrix, torch, normalize=args.pca_normalize)
    pca_layer_rows = [
        {
            "size": label,
            "layer": layer,
            "pc1": layer_coords[idx, 0].item(),
            "pc2": layer_coords[idx, 1].item(),
            "explained_pc1": layer_explained[0],
            "explained_pc2": layer_explained[1],
            "pca_normalized_vectors": args.pca_normalize,
        }
        for idx, label in enumerate(labels)
    ]
    write_rows(
        out_dir / f"pca_layer{layer}.csv",
        [
            "size",
            "layer",
            "pc1",
            "pc2",
            "explained_pc1",
            "explained_pc2",
            "pca_normalized_vectors",
        ],
        pca_layer_rows,
    )

    all_layer_matrices = []
    all_layer_index: list[tuple[str, int]] = []
    for label in labels:
        for current_layer in range(num_layers):
            all_layer_matrices.append(vectors[label][current_layer])
            all_layer_index.append((label, current_layer))

    all_matrix = torch.stack(all_layer_matrices, dim=0)
    all_coords, all_explained = run_pca(all_matrix, torch, normalize=args.pca_normalize)
    pca_all_rows = []
    for idx, (label, current_layer) in enumerate(all_layer_index):
        pca_all_rows.append(
            {
                "size": label,
                "layer": current_layer,
                "pc1": all_coords[idx, 0].item(),
                "pc2": all_coords[idx, 1].item(),
                "explained_pc1": all_explained[0],
                "explained_pc2": all_explained[1],
                "pca_normalized_vectors": args.pca_normalize,
            }
        )
    write_rows(
        out_dir / "pca_all_layers.csv",
        [
            "size",
            "layer",
            "pc1",
            "pc2",
            "explained_pc1",
            "explained_pc2",
            "pca_normalized_vectors",
        ],
        pca_all_rows,
    )

    write_summary(
        out_dir / "summary.txt",
        args,
        labels,
        layer,
        reference_size,
        vector_paths,
        summary_rows,
        adjacent_rows,
    )

    if not args.no_plots:
        plt = maybe_import_matplotlib()
        if plt is None:
            print("matplotlib is not installed; wrote CSV/text outputs but skipped plots.", file=sys.stderr)
        else:
            plot_pairwise_heatmap(
                plt,
                out_dir / f"pairwise_cosine_layer{layer}_heatmap.png",
                labels,
                pairwise_layer,
                f"Pairwise cosine similarity, layer {layer}",
                args.dpi,
            )

            plot_lines_by_layer(
                plt,
                out_dir / "cosine_to_reference_by_layer.png",
                reference_rows,
                group_key="size",
                value_key="cosine",
                title=f"Cosine similarity to {reference_size} by layer",
                ylabel=f"Cosine to {reference_size}",
                dpi=args.dpi,
            )

            layer_ref_values = [
                float(row["cosine_to_reference_layer"]) for row in summary_rows
            ]
            plot_size_series(
                plt,
                out_dir / f"cosine_to_{reference_size}_layer{layer}.png",
                labels,
                layer_ref_values,
                f"Cosine similarity to {reference_size}, layer {layer}",
                f"Cosine to {reference_size}",
                args.dpi,
            )

            plot_lines_by_layer(
                plt,
                out_dir / "adjacent_size_cosine_by_layer.png",
                adjacent_rows,
                group_key="comparison",
                value_key="cosine",
                title="Adjacent sample-size cosine similarity by layer",
                ylabel="Cosine similarity",
                dpi=args.dpi,
            )

            plot_lines_by_layer(
                plt,
                out_dir / "vector_norms_by_layer.png",
                norm_rows,
                group_key="size",
                value_key="l2_norm",
                title="Steering-vector L2 norm by layer",
                ylabel="L2 norm",
                dpi=args.dpi,
            )

            pca_mode = "normalized" if args.pca_normalize else "raw"
            plot_pca_layer(
                plt,
                out_dir / f"pca_layer{layer}.png",
                labels,
                layer_coords,
                layer_explained,
                f"PCA trajectory by sample size, layer {layer} ({pca_mode} vectors)",
                args.dpi,
            )

            plot_pca_all_layers(
                plt,
                out_dir / "pca_all_layers.png",
                pca_all_rows,
                labels,
                selected_layer=layer,
                explained=all_explained,
                title=f"PCA across all layers and sample sizes ({pca_mode} vectors)",
                dpi=args.dpi,
            )

            if args.plot_each_layer_pca:
                pca_layers_dir = out_dir / "pca_by_layer"
                pca_layers_dir.mkdir(parents=True, exist_ok=True)
                for current_layer in range(num_layers):
                    matrix = torch.stack([vectors[label][current_layer] for label in labels], dim=0)
                    coords, explained = run_pca(matrix, torch, normalize=args.pca_normalize)
                    plot_pca_layer(
                        plt,
                        pca_layers_dir / f"pca_layer{current_layer}.png",
                        labels,
                        coords,
                        explained,
                        f"PCA trajectory by sample size, layer {current_layer} ({pca_mode} vectors)",
                        args.dpi,
                    )

    print("Wrote:")
    for filename in [
        "summary.txt",
        "summary.csv",
        f"pairwise_cosine_layer{layer}.csv",
        "cosine_to_reference_by_layer.csv",
        "adjacent_size_cosine_by_layer.csv",
        "vector_norms_by_layer.csv",
        f"pca_layer{layer}.csv",
        "pca_all_layers.csv",
    ]:
        print(f"  {out_dir / filename}")


if __name__ == "__main__":
    main()
