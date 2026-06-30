#!/usr/bin/env python3
"""
Create paper-ready residual-projection coverage figures.

Default usage from the repository root:

  python neuron_steering/plot_lape_residual_projection_figures.py

This reads:

  neuron_steering/coverage_output/all_dialects_lape_residual_projection/

and writes PNG and PDF figures suitable for a paper submission:

  - residual_projection_global_coverage.{png,pdf}
  - residual_projection_coverage_vs_budget.{png,pdf}
  - residual_projection_layer_heatmap.{png,pdf}
  - residual_projection_layer_mean_trend.{png,pdf}

The script only reads saved CSV outputs; it does not load model weights.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires matplotlib. Install it with: pip install matplotlib"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "coverage_output" / "all_dialects_lape_residual_projection"
DEFAULT_RUN_NAMES = (
    ("ALLaM-7B-Instruct-preview", "allam_7dialects"),
    ("Fanar-1-9B-Instruct", "fanar_instruct_7dialects"),
)

DIALECT_NAMES = {
    "CAI": "Cairo",
    "RAB": "Rabat",
    "DOH": "Doha",
    "RIY": "Riyadh",
    "ALE": "Aleppo",
    "BEI": "Beirut",
    "DAM": "Damascus",
    "JED": "Jeddah",
    "TUN": "Tunis",
    "KHA": "Khartoum",
}
DEFAULT_DIALECT_ORDER = ("CAI", "RAB", "DOH", "RIY", "ALE", "BEI", "DAM", "JED", "TUN", "KHA")

MODEL_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
MODEL_MARKERS = ("o", "s", "^", "D")
GRID_COLOR = "#D8D8D8"


@dataclass(frozen=True)
class SummaryRecord:
    model: str
    run_slug: str
    dialect: str
    dialect_name: str
    model_id: str
    vector_path: str
    output_dir: Path
    selected_neurons: int
    selected_dimension_fraction_pct: float
    total_energy: float
    projected_energy: float
    coverage_percent: float


@dataclass(frozen=True)
class LayerRecord:
    model: str
    run_slug: str
    dialect: str
    dialect_name: str
    layer: int
    vector_layer: int
    selected_neurons: int
    subspace_rank: int
    layer_energy: float
    projected_energy: float
    coverage_percent: float
    selected_dimension_fraction_pct: float


@dataclass
class RunData:
    label: str
    slug: str
    input_dir: Path
    summaries: Dict[str, SummaryRecord]
    layers_by_dialect: Dict[str, List[LayerRecord]]
    dialect_order: List[str]
    num_layers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create EMNLP-style residual-projection coverage plots from "
            "measure_lape_residual_projection.py outputs."
        )
    )
    parser.add_argument(
        "--input_root",
        "--input-root",
        dest="input_root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root output directory from the residual-projection coverage runs.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="LABEL=DIR",
        help=(
            "Run directory to plot. Can be passed multiple times. "
            "Default: ALLaM and Fanar Instruct under --input_root."
        ),
    )
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Directory where figures and plot-data CSVs are written. Default: <input_root>/paper_figures.",
    )
    parser.add_argument(
        "--prefix",
        default="residual_projection",
        help="Filename prefix for generated figures and CSVs.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated figure formats to save, e.g. png,pdf.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="DPI for raster formats.")
    parser.add_argument(
        "--dialect_order",
        "--dialect-order",
        dest="dialect_order",
        default=",".join(DEFAULT_DIALECT_ORDER),
        help="Comma- or space-separated preferred dialect code order.",
    )
    parser.add_argument(
        "--heatmap_vmax",
        "--heatmap-vmax",
        dest="heatmap_vmax",
        type=float,
        default=None,
        help="Optional maximum value for heatmap color scale. Default: observed maximum.",
    )
    parser.add_argument(
        "--max_x_ticks",
        "--max-x-ticks",
        dest="max_x_ticks",
        type=int,
        default=8,
        help="Maximum labeled layer ticks on heatmaps.",
    )
    parser.add_argument(
        "--no_titles",
        "--no-titles",
        dest="no_titles",
        action="store_true",
        help="Omit figure titles for camera-ready layouts where captions carry the title.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "run"


def parse_formats(raw: str) -> List[str]:
    formats = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not formats:
        return ["png"]
    allowed = {"png", "pdf", "svg", "jpg", "jpeg"}
    invalid = sorted(set(formats) - allowed)
    if invalid:
        raise ValueError(f"Unsupported output format(s): {', '.join(invalid)}")
    return formats


def parse_dialect_order(raw: str) -> List[str]:
    parts = [part.strip() for part in re.split(r"[,\s]+", raw) if part.strip()]
    return parts or list(DEFAULT_DIALECT_ORDER)


def parse_run_spec(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Invalid --run value {value!r}. Expected LABEL=DIR.")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return label, path.resolve()


def to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def vector_dialect_name(row: Dict[str, str]) -> str:
    dialect = str(row.get("target_dialect", "")).strip()
    vector_path = str(row.get("vector_path", "")).strip()
    if vector_path:
        name = Path(vector_path).name
        suffix = "_response_avg_diff.pt"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return DIALECT_NAMES.get(dialect, dialect)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summary_files_for_run(run_dir: Path) -> List[Path]:
    aggregate = run_dir / "residual_projection_summary_all_dialects.csv"
    if aggregate.exists():
        return [aggregate]
    return sorted(run_dir.glob("*/residual_projection_summary.csv"))


def make_summary_record(label: str, slug: str, row: Dict[str, str], source_path: Path) -> Optional[SummaryRecord]:
    if str(row.get("status", "")).strip() != "ok":
        return None

    dialect = str(row.get("target_dialect", "")).strip()
    if not dialect:
        return None

    output_dir_text = str(row.get("output_dir", "")).strip()
    output_dir = Path(output_dir_text) if output_dir_text else source_path.parent
    if not output_dir.is_absolute():
        output_dir = (source_path.parent / output_dir).resolve()

    selected_fraction = to_float(row.get("selected_dimension_fraction")) * 100.0
    return SummaryRecord(
        model=label,
        run_slug=slug,
        dialect=dialect,
        dialect_name=vector_dialect_name(row),
        model_id=str(row.get("model_id", "")).strip(),
        vector_path=str(row.get("vector_path", "")).strip(),
        output_dir=output_dir,
        selected_neurons=to_int(row.get("selected_neurons")),
        selected_dimension_fraction_pct=selected_fraction,
        total_energy=to_float(row.get("total_energy")),
        projected_energy=to_float(row.get("projected_energy")),
        coverage_percent=to_float(row.get("coverage_percent")),
    )


def make_layer_record(
    summary: SummaryRecord,
    row: Dict[str, str],
) -> LayerRecord:
    return LayerRecord(
        model=summary.model,
        run_slug=summary.run_slug,
        dialect=summary.dialect,
        dialect_name=summary.dialect_name,
        layer=to_int(row.get("layer")),
        vector_layer=to_int(row.get("vector_layer")),
        selected_neurons=to_int(row.get("selected_neurons")),
        subspace_rank=to_int(row.get("subspace_rank")),
        layer_energy=to_float(row.get("layer_energy")),
        projected_energy=to_float(row.get("projected_energy")),
        coverage_percent=to_float(row.get("coverage_percent")),
        selected_dimension_fraction_pct=to_float(row.get("selected_dimension_fraction")) * 100.0,
    )


def order_dialects(dialects: Iterable[str], preferred_order: Sequence[str]) -> List[str]:
    dialects_list = list(dict.fromkeys(dialects))
    rank = {dialect: idx for idx, dialect in enumerate(preferred_order)}
    return sorted(dialects_list, key=lambda d: (rank.get(d, len(rank)), d))


def load_run(label: str, input_dir: Path, preferred_order: Sequence[str]) -> RunData:
    if not input_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {input_dir}")

    slug = slugify(label)
    summaries: Dict[str, SummaryRecord] = {}

    for summary_path in summary_files_for_run(input_dir):
        for row in read_csv_rows(summary_path):
            record = make_summary_record(label, slug, row, summary_path)
            if record is not None:
                summaries[record.dialect] = record

    if not summaries:
        raise ValueError(f"No successful residual-projection summaries found in {input_dir}")

    layers_by_dialect: Dict[str, List[LayerRecord]] = {}
    max_layer = -1
    for dialect, summary in summaries.items():
        layer_path = summary.output_dir / "residual_projection_by_layer.csv"
        if not layer_path.exists():
            print(f"warning: missing layer CSV for {label} {dialect}: {layer_path}", file=sys.stderr)
            layers_by_dialect[dialect] = []
            continue
        rows = [make_layer_record(summary, row) for row in read_csv_rows(layer_path)]
        rows.sort(key=lambda row: row.layer)
        layers_by_dialect[dialect] = rows
        if rows:
            max_layer = max(max_layer, max(row.layer for row in rows))

    dialect_order = order_dialects(summaries.keys(), preferred_order)
    return RunData(
        label=label,
        slug=slug,
        input_dir=input_dir,
        summaries=summaries,
        layers_by_dialect=layers_by_dialect,
        dialect_order=dialect_order,
        num_layers=max_layer + 1,
    )


def configure_matplotlib(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.7,
            "legend.frameon": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.82,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: object, out_base: Path, formats: Sequence[str], dpi: int) -> List[Path]:
    written: List[Path] = []
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        kwargs = {"format": fmt}
        if fmt in {"png", "jpg", "jpeg"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def all_dialects(runs: Sequence[RunData], preferred_order: Sequence[str]) -> List[str]:
    dialects: List[str] = []
    for run in runs:
        dialects.extend(run.dialect_order)
    return order_dialects(dialects, preferred_order)


def dialect_label(dialect: str, runs: Sequence[RunData], include_code: bool = False) -> str:
    for run in runs:
        summary = run.summaries.get(dialect)
        if summary is not None:
            if include_code and summary.dialect_name != dialect:
                return f"{summary.dialect_name}\n({dialect})"
            return summary.dialect_name
    return DIALECT_NAMES.get(dialect, dialect)


def nice_upper(values: Sequence[float], floor: float = 1.0, pad: float = 1.12) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return floor
    raw = max(max(finite) * pad, floor)
    if raw <= 10:
        step = 1
    elif raw <= 25:
        step = 2.5
    elif raw <= 60:
        step = 5
    else:
        step = 10
    return math.ceil(raw / step) * step


def nice_layer_ticks(num_layers: int, max_ticks: int) -> List[int]:
    if num_layers <= 0:
        return []
    max_ticks = max(2, max_ticks)
    step = max(1, math.ceil(num_layers / max_ticks))
    ticks = list(range(0, num_layers, step))
    if ticks[-1] != num_layers - 1:
        ticks.append(num_layers - 1)
    return ticks


def plot_global_coverage(
    runs: Sequence[RunData],
    dialects: Sequence[str],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    show_title: bool,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    x_positions = list(range(len(dialects)))
    bar_width = min(0.34, 0.78 / max(1, len(runs)))
    offsets = [
        (idx - (len(runs) - 1) / 2.0) * bar_width
        for idx in range(len(runs))
    ]

    all_values: List[float] = []
    for run_idx, run in enumerate(runs):
        values = [
            run.summaries[dialect].coverage_percent if dialect in run.summaries else math.nan
            for dialect in dialects
        ]
        all_values.extend(v for v in values if math.isfinite(v))
        xs = [x + offsets[run_idx] for x in x_positions]
        bars = ax.bar(
            xs,
            values,
            width=bar_width * 0.92,
            color=MODEL_COLORS[run_idx % len(MODEL_COLORS)],
            edgecolor="white",
            linewidth=0.7,
            label=run.label,
        )
        for value, bar in zip(values, bars):
            if not math.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.35,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                rotation=0,
            )

    ax.set_ylabel("Residual-subspace coverage (%)")
    ax.set_xlabel("Target dialect")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([dialect_label(d, runs) for d in dialects], rotation=20, ha="right")
    ax.set_ylim(0, nice_upper(all_values, floor=10.0))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", ncol=max(1, len(runs)))
    if show_title:
        ax.set_title("Residual Projection Coverage by Dialect")
    fig.tight_layout()
    return save_figure(fig, out_base, formats, dpi)


def plot_coverage_vs_budget(
    runs: Sequence[RunData],
    dialects: Sequence[str],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    show_title: bool,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(6.65, 4.25))
    x_values: List[float] = []
    y_values: List[float] = []

    for run_idx, run in enumerate(runs):
        color = MODEL_COLORS[run_idx % len(MODEL_COLORS)]
        marker = MODEL_MARKERS[run_idx % len(MODEL_MARKERS)]
        first = True
        for dialect_idx, dialect in enumerate(dialects):
            summary = run.summaries.get(dialect)
            if summary is None:
                continue
            x = summary.selected_dimension_fraction_pct
            y = summary.coverage_percent
            x_values.append(x)
            y_values.append(y)
            ax.scatter(
                [x],
                [y],
                s=70,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=run.label if first else None,
                zorder=3,
            )
            first = False
            y_offset = 0.55 if dialect_idx % 2 == 0 else -0.85
            ax.annotate(
                dialect,
                xy=(x, y),
                xytext=(4, y_offset),
                textcoords="offset points",
                fontsize=8.1,
                color="#202020",
            )

    ax.set_xlabel("Selected neurons (% of all MLP dimensions)")
    ax.set_ylabel("Residual-subspace coverage (%)")
    ax.set_xlim(0, nice_upper(x_values, floor=0.2, pad=1.18))
    ax.set_ylim(0, nice_upper(y_values, floor=10.0, pad=1.15))
    ax.grid(True, axis="both")
    ax.legend(loc="upper left")
    if show_title:
        ax.set_title("Coverage Relative to Selected-Neuron Budget")
    fig.tight_layout()
    return save_figure(fig, out_base, formats, dpi)


def coverage_matrix(run: RunData, dialects: Sequence[str]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for dialect in dialects:
        by_layer = {row.layer: row.coverage_percent for row in run.layers_by_dialect.get(dialect, [])}
        matrix.append([by_layer.get(layer, math.nan) for layer in range(run.num_layers)])
    return matrix


def all_layer_values(runs: Sequence[RunData]) -> List[float]:
    values: List[float] = []
    for run in runs:
        for rows in run.layers_by_dialect.values():
            values.extend(row.coverage_percent for row in rows if math.isfinite(row.coverage_percent))
    return values


def plot_layer_heatmap(
    runs: Sequence[RunData],
    dialects: Sequence[str],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    heatmap_vmax: Optional[float],
    max_x_ticks: int,
    show_title: bool,
) -> List[Path]:
    values = all_layer_values(runs)
    vmax = heatmap_vmax if heatmap_vmax is not None else (max(values) if values else 1.0)
    vmax = max(vmax, 1.0)

    fig_height = max(3.2, 1.95 * len(runs) + 0.75)
    fig, axs = plt.subplots(
        len(runs),
        1,
        figsize=(7.35, fig_height),
        sharex=False,
        squeeze=False,
        constrained_layout=True,
    )
    axes = [row[0] for row in axs]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#F2F2F2")
    image = None

    for ax, run in zip(axes, runs):
        matrix = coverage_matrix(run, dialects)
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax, cmap=cmap)
        ax.set_title(run.label, loc="left", fontweight="bold")
        ax.set_ylabel("Dialect")
        ax.set_yticks(range(len(dialects)))
        ax.set_yticklabels([dialect_label(d, runs) for d in dialects])
        ticks = nice_layer_ticks(run.num_layers, max_x_ticks)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(tick) for tick in ticks])
        ax.set_xlabel("Layer")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.92, pad=0.018)
        cbar.set_label("Layer coverage (%)")

    if show_title:
        fig.suptitle("Layer-wise Residual Projection Coverage", fontsize=11.8, fontweight="bold")
    return save_figure(fig, out_base, formats, dpi)


def layer_weighted_stats(run: RunData) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs: List[float] = []
    means: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    if run.num_layers <= 0:
        return xs, means, lows, highs

    for layer in range(run.num_layers):
        layer_rows = [
            rows[layer]
            for rows in run.layers_by_dialect.values()
            if len(rows) > layer and rows[layer].layer == layer
        ]
        if not layer_rows:
            continue
        total_energy = sum(row.layer_energy for row in layer_rows)
        projected_energy = sum(row.projected_energy for row in layer_rows)
        weighted = 100.0 * projected_energy / total_energy if total_energy > 0 else 0.0
        coverages = [row.coverage_percent for row in layer_rows]
        x = layer / (run.num_layers - 1) if run.num_layers > 1 else 0.0
        xs.append(x)
        means.append(weighted)
        lows.append(min(coverages))
        highs.append(max(coverages))
    return xs, means, lows, highs


def plot_layer_mean_trend(
    runs: Sequence[RunData],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    show_title: bool,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    y_values: List[float] = []

    for run_idx, run in enumerate(runs):
        xs, means, lows, highs = layer_weighted_stats(run)
        if not xs:
            continue
        color = MODEL_COLORS[run_idx % len(MODEL_COLORS)]
        y_values.extend(highs)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.13, linewidth=0)
        ax.plot(
            xs,
            means,
            color=color,
            linewidth=2.2,
            marker=MODEL_MARKERS[run_idx % len(MODEL_MARKERS)],
            markersize=3.4,
            markevery=max(1, len(xs) // 8),
            label=run.label,
        )

    ax.set_xlabel("Relative layer depth")
    ax.set_ylabel("Coverage (%)")
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(0, nice_upper(y_values, floor=10.0, pad=1.12))
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.grid(True, axis="both")
    ax.legend(loc="upper left")
    if show_title:
        ax.set_title("Layer Trend of Projection Coverage")
    fig.tight_layout()
    return save_figure(fig, out_base, formats, dpi)


def write_summary_plot_data(path: Path, runs: Sequence[RunData], dialects: Sequence[str]) -> None:
    fieldnames = [
        "model",
        "dialect",
        "dialect_name",
        "coverage_percent",
        "selected_neurons",
        "selected_dimension_fraction_percent",
        "coverage_per_selected_percent",
        "total_energy",
        "projected_energy",
        "model_id",
        "vector_path",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for dialect in dialects:
                summary = run.summaries.get(dialect)
                if summary is None:
                    continue
                efficiency = (
                    summary.coverage_percent / summary.selected_dimension_fraction_pct
                    if summary.selected_dimension_fraction_pct > 0
                    else 0.0
                )
                writer.writerow(
                    {
                        "model": summary.model,
                        "dialect": summary.dialect,
                        "dialect_name": summary.dialect_name,
                        "coverage_percent": f"{summary.coverage_percent:.10f}",
                        "selected_neurons": summary.selected_neurons,
                        "selected_dimension_fraction_percent": f"{summary.selected_dimension_fraction_pct:.10f}",
                        "coverage_per_selected_percent": f"{efficiency:.10f}",
                        "total_energy": f"{summary.total_energy:.10f}",
                        "projected_energy": f"{summary.projected_energy:.10f}",
                        "model_id": summary.model_id,
                        "vector_path": summary.vector_path,
                        "output_dir": str(summary.output_dir),
                    }
                )


def write_layer_plot_data(path: Path, runs: Sequence[RunData], dialects: Sequence[str]) -> None:
    fieldnames = [
        "model",
        "dialect",
        "dialect_name",
        "layer",
        "relative_layer_depth",
        "vector_layer",
        "coverage_percent",
        "selected_neurons",
        "subspace_rank",
        "layer_energy",
        "projected_energy",
        "selected_dimension_fraction_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            denom = run.num_layers - 1 if run.num_layers > 1 else 1
            for dialect in dialects:
                for row in run.layers_by_dialect.get(dialect, []):
                    writer.writerow(
                        {
                            "model": row.model,
                            "dialect": row.dialect,
                            "dialect_name": row.dialect_name,
                            "layer": row.layer,
                            "relative_layer_depth": f"{row.layer / denom:.10f}",
                            "vector_layer": row.vector_layer,
                            "coverage_percent": f"{row.coverage_percent:.10f}",
                            "selected_neurons": row.selected_neurons,
                            "subspace_rank": row.subspace_rank,
                            "layer_energy": f"{row.layer_energy:.10f}",
                            "projected_energy": f"{row.projected_energy:.10f}",
                            "selected_dimension_fraction_percent": f"{row.selected_dimension_fraction_pct:.10f}",
                        }
                    )


def log_run_summary(runs: Sequence[RunData]) -> None:
    for run in runs:
        coverages = [summary.coverage_percent for summary in run.summaries.values()]
        mean_cov = sum(coverages) / len(coverages) if coverages else 0.0
        best = max(run.summaries.values(), key=lambda item: item.coverage_percent)
        print(
            f"{run.label}: {len(run.summaries)} dialects, mean coverage={mean_cov:.2f}%, "
            f"max={best.dialect}/{best.dialect_name} {best.coverage_percent:.2f}%",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)
    preferred_order = parse_dialect_order(args.dialect_order)
    configure_matplotlib(args.dpi)

    input_root = Path(args.input_root).expanduser()
    if not input_root.is_absolute():
        input_root = Path.cwd() / input_root
    input_root = input_root.resolve()

    if args.run:
        run_specs = [parse_run_spec(value) for value in args.run]
    else:
        run_specs = [(label, (input_root / dirname).resolve()) for label, dirname in DEFAULT_RUN_NAMES]

    runs = [load_run(label, run_dir, preferred_order) for label, run_dir in run_specs]
    dialects = all_dialects(runs, preferred_order)

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else input_root / "paper_figures"
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_run_summary(runs)

    show_title = not args.no_titles
    written: List[Path] = []
    written.extend(
        plot_global_coverage(
            runs=runs,
            dialects=dialects,
            out_base=out_dir / f"{args.prefix}_global_coverage",
            formats=formats,
            dpi=args.dpi,
            show_title=show_title,
        )
    )
    written.extend(
        plot_coverage_vs_budget(
            runs=runs,
            dialects=dialects,
            out_base=out_dir / f"{args.prefix}_coverage_vs_budget",
            formats=formats,
            dpi=args.dpi,
            show_title=show_title,
        )
    )
    written.extend(
        plot_layer_heatmap(
            runs=runs,
            dialects=dialects,
            out_base=out_dir / f"{args.prefix}_layer_heatmap",
            formats=formats,
            dpi=args.dpi,
            heatmap_vmax=args.heatmap_vmax,
            max_x_ticks=args.max_x_ticks,
            show_title=show_title,
        )
    )
    written.extend(
        plot_layer_mean_trend(
            runs=runs,
            out_base=out_dir / f"{args.prefix}_layer_mean_trend",
            formats=formats,
            dpi=args.dpi,
            show_title=show_title,
        )
    )

    summary_table = out_dir / f"{args.prefix}_summary_plot_data.csv"
    layer_table = out_dir / f"{args.prefix}_layer_plot_data.csv"
    write_summary_plot_data(summary_table, runs, dialects)
    write_layer_plot_data(layer_table, runs, dialects)

    print("Wrote figures:", file=sys.stderr)
    for path in written:
        print(f"  {path}", file=sys.stderr)
    print("Wrote plot-data tables:", file=sys.stderr)
    print(f"  {summary_table}", file=sys.stderr)
    print(f"  {layer_table}", file=sys.stderr)


if __name__ == "__main__":
    main()
