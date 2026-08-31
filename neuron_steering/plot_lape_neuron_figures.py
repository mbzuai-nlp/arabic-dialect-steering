#!/usr/bin/env python3
"""
Create paper-ready LAPE identified-neuron figures for extraction outputs.

Default usage from the repository root:

  python neuron_steering/plot_lape_neuron_figures.py

This reads the default ALLaM and Fanar LAPE extraction directories and writes:
  - identified_neurons_layer_trend.{png,pdf}
  - identified_neurons_layer_bar.{png,pdf}
  - identified_neurons_overlap_jaccard.{png,pdf}
  - CSV tables with the plotted counts and overlap values

The script only reads saved extraction artifacts; it does not run any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires matplotlib. Install it with: pip install matplotlib"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "extraction_output"
DEFAULT_RUNS = (
    ("ALLaM", DEFAULT_OUTPUT_ROOT / "lape_allam_lape1p0_act95p0_no-special"),
    ("Fanar", DEFAULT_OUTPUT_ROOT / "lape_fanar_lape1p0_act95p0_no-special"),
)

COLOR_CYCLE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
TOTAL_COLOR = "#2F2F2F"


@dataclass(frozen=True)
class SelectedNeuronRecord:
    dialect: str
    layer: int
    neuron: int


@dataclass
class RunData:
    label: str
    slug: str
    input_dir: Path
    dialects: List[str]
    records: List[SelectedNeuronRecord]
    sets_by_dialect: Dict[str, set[Tuple[int, int]]]
    layer_counts_by_dialect: Dict[str, Dict[int, int]]
    layer_unique_counts: Dict[int, int]
    layer_record_counts: Dict[int, int]
    num_layers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-ready layer trend and dialect-overlap plots from "
            "LAPE selected_neurons.csv extraction outputs."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="LABEL=DIR",
        help=(
            "Extraction run to plot. Can be passed multiple times. "
            "Default: ALLaM and Fanar extraction outputs."
        ),
    )
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        dest="out_dir",
        default=str(DEFAULT_OUTPUT_ROOT / "paper_figures"),
        help="Directory where figures and CSV tables will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="identified_neurons",
        help="Filename prefix for generated figures and tables.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated figure formats to save, e.g. png,pdf.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster formats.")
    parser.add_argument(
        "--max_x_ticks",
        "--max-x-ticks",
        dest="max_x_ticks",
        type=int,
        default=8,
        help="Maximum labeled layer ticks on each trend/bar subplot.",
    )
    parser.add_argument(
        "--trend_title",
        "--trend-title",
        dest="trend_title",
        default="Layer-wise Counts of Identified Neurons",
        help="Title for the layer trend figure.",
    )
    parser.add_argument(
        "--bar_title",
        "--bar-title",
        dest="bar_title",
        default="Layer-wise Identified-Neuron Counts by Dialect",
        help="Title for the per-layer bar figure.",
    )
    parser.add_argument(
        "--overlap_title",
        "--overlap-title",
        dest="overlap_title",
        default="Overlap of Identified Neuron Sets Across Dialects",
        help="Title for the overlap heatmap figure.",
    )
    parser.add_argument(
        "--no_total",
        "--no-total",
        dest="no_total",
        action="store_true",
        help="Do not draw the total-unique-neurons trend line.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "run"


def parse_run_spec(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --run value {value!r}. Expected LABEL=DIR."
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return label, path.resolve()


def parse_formats(raw: str) -> List[str]:
    formats = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not formats:
        return ["png"]
    allowed = {"png", "pdf", "svg", "jpg", "jpeg"}
    invalid = sorted(set(formats) - allowed)
    if invalid:
        raise ValueError(f"Unsupported output format(s): {', '.join(invalid)}")
    return formats


def read_json(path: Path) -> object | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_dialects(input_dir: Path, records: Sequence[SelectedNeuronRecord]) -> List[str]:
    run_summary = read_json(input_dir / "run_summary.json")
    if isinstance(run_summary, dict) and isinstance(run_summary.get("dialects"), list):
        return [str(d) for d in run_summary["dialects"]]

    dialects_json = read_json(input_dir / "dialects.json")
    if isinstance(dialects_json, list):
        return [str(d) for d in dialects_json]
    if isinstance(dialects_json, dict):
        try:
            return [
                str(value)
                for _, value in sorted(
                    ((int(key), value) for key, value in dialects_json.items()),
                    key=lambda item: item[0],
                )
            ]
        except Exception:
            return [str(value) for value in dialects_json.values()]

    dialects: List[str] = []
    seen = set()
    for record in records:
        if record.dialect not in seen:
            seen.add(record.dialect)
            dialects.append(record.dialect)
    return dialects


def load_selected_records(input_dir: Path) -> List[SelectedNeuronRecord]:
    selected_path = input_dir / "selected_neurons.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected-neuron table: {selected_path}")

    records: List[SelectedNeuronRecord] = []
    with selected_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dialect", "layer", "neuron"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{selected_path} is missing required column(s): {', '.join(sorted(missing))}"
            )
        for row in reader:
            try:
                records.append(
                    SelectedNeuronRecord(
                        dialect=str(row["dialect"]),
                        layer=int(float(row["layer"])),
                        neuron=int(float(row["neuron"])),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid selected-neuron row in {selected_path}: {row}") from exc

    if not records:
        raise ValueError(f"No selected-neuron records found in {selected_path}")
    return records


def load_run(label: str, input_dir: Path) -> RunData:
    records = load_selected_records(input_dir)
    dialects = infer_dialects(input_dir, records)
    if not dialects:
        raise ValueError(f"Could not infer dialect order for {input_dir}")

    sets_by_dialect: Dict[str, set[Tuple[int, int]]] = {dialect: set() for dialect in dialects}
    layer_counts_by_dialect: Dict[str, Dict[int, int]] = {dialect: {} for dialect in dialects}
    layer_unique_sets: Dict[int, set[Tuple[int, int]]] = {}
    layer_record_counts: Dict[int, int] = {}

    for record in records:
        sets_by_dialect.setdefault(record.dialect, set()).add((record.layer, record.neuron))
        dialect_counts = layer_counts_by_dialect.setdefault(record.dialect, {})
        dialect_counts[record.layer] = dialect_counts.get(record.layer, 0) + 1
        layer_unique_sets.setdefault(record.layer, set()).add((record.layer, record.neuron))
        layer_record_counts[record.layer] = layer_record_counts.get(record.layer, 0) + 1

    for record in records:
        if record.dialect not in dialects:
            dialects.append(record.dialect)

    max_layer = max(record.layer for record in records)
    run_summary = read_json(input_dir / "run_summary.json")
    if isinstance(run_summary, dict) and run_summary.get("num_layers") is not None:
        try:
            num_layers = max(int(run_summary["num_layers"]), max_layer + 1)
        except (TypeError, ValueError):
            num_layers = max_layer + 1
    else:
        num_layers = max_layer + 1

    layer_unique_counts = {
        layer: len(layer_unique_sets.get(layer, set())) for layer in range(num_layers)
    }
    layer_record_counts = {
        layer: layer_record_counts.get(layer, 0) for layer in range(num_layers)
    }
    for dialect in dialects:
        counts = layer_counts_by_dialect.setdefault(dialect, {})
        for layer in range(num_layers):
            counts.setdefault(layer, 0)

    return RunData(
        label=label,
        slug=slugify(label),
        input_dir=input_dir,
        dialects=dialects,
        records=records,
        sets_by_dialect=sets_by_dialect,
        layer_counts_by_dialect=layer_counts_by_dialect,
        layer_unique_counts=layer_unique_counts,
        layer_record_counts=layer_record_counts,
        num_layers=num_layers,
    )


def configure_matplotlib(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "lines.linewidth": 2.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: object, out_base: Path, formats: Sequence[str], dpi: int) -> List[Path]:
    written: List[Path] = []
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        save_kwargs = {"format": fmt}
        if fmt in {"png", "jpg", "jpeg"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def collect_legend(axs: Iterable[object]) -> Tuple[List[object], List[str]]:
    handles_by_label: Dict[str, object] = {}
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    labels = list(handles_by_label.keys())
    handles = [handles_by_label[label] for label in labels]
    return handles, labels


def nice_layer_ticks(num_layers: int, max_ticks: int) -> List[int]:
    if num_layers <= 0:
        return []
    max_ticks = max(2, max_ticks)
    step = max(1, math.ceil(num_layers / max_ticks))
    ticks = list(range(0, num_layers, step))
    if ticks[-1] != num_layers - 1:
        ticks.append(num_layers - 1)
    return ticks


def set_layer_x_axis(ax: object, num_layers: int, max_ticks: int) -> None:
    ticks = nice_layer_ticks(num_layers, max_ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks], rotation=0, ha="center")
    ax.set_xlim(-0.6, num_layers - 0.4)


def plot_layer_trend(
    runs: Sequence[RunData],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    title: str,
    show_total: bool,
    max_x_ticks: int,
) -> List[Path]:
    y_max = 0
    for run in runs:
        if show_total:
            y_max = max(y_max, max(run.layer_unique_counts.values(), default=0))
        for dialect in run.dialects:
            y_max = max(y_max, max(run.layer_counts_by_dialect[dialect].values(), default=0))

    fig_width = max(7.2, 5.25 * len(runs))
    fig, axs = plt.subplots(
        1,
        len(runs),
        figsize=(fig_width, 4.6),
        sharey=True,
        squeeze=False,
    )
    axes = list(axs[0])
    for ax, run in zip(axes, runs):
        layers = list(range(run.num_layers))
        for index, dialect in enumerate(run.dialects):
            counts = [run.layer_counts_by_dialect[dialect].get(layer, 0) for layer in layers]
            ax.plot(
                layers,
                counts,
                marker="o",
                markersize=3.7,
                color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
                label=dialect,
            )
        if show_total:
            total_counts = [run.layer_unique_counts.get(layer, 0) for layer in layers]
            ax.plot(
                layers,
                total_counts,
                marker="s",
                markersize=3.4,
                color=TOTAL_COLOR,
                linestyle="--",
                linewidth=2.4,
                label="Total unique",
            )
        ax.set_title(run.label)
        ax.set_xlabel("Layer")
        set_layer_x_axis(ax, run.num_layers, max_x_ticks)
        ax.grid(True, axis="y")
        ax.grid(True, axis="x", alpha=0.28)
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.12)

    axes[0].set_ylabel("Identified neurons (count)")
    handles, labels = collect_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(max(1, len(labels)), 6),
        )
    fig.suptitle(title, y=1.10, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, out_base, formats, dpi)


def plot_layer_bars(
    runs: Sequence[RunData],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    title: str,
    max_x_ticks: int,
) -> List[Path]:
    y_max = 0
    for run in runs:
        y_max = max(y_max, max(run.layer_record_counts.values(), default=0))

    fig_width = max(7.2, 5.25 * len(runs))
    fig, axs = plt.subplots(
        1,
        len(runs),
        figsize=(fig_width, 4.8),
        sharey=True,
        squeeze=False,
    )
    axes = list(axs[0])
    for ax, run in zip(axes, runs):
        layers = list(range(run.num_layers))
        bottom = [0 for _ in layers]
        for index, dialect in enumerate(run.dialects):
            counts = [run.layer_counts_by_dialect[dialect].get(layer, 0) for layer in layers]
            ax.bar(
                layers,
                counts,
                bottom=bottom,
                width=0.84,
                color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
                edgecolor="white",
                linewidth=0.25,
                label=dialect,
            )
            bottom = [base + value for base, value in zip(bottom, counts)]
        ax.set_title(run.label)
        ax.set_xlabel("Layer")
        set_layer_x_axis(ax, run.num_layers, max_x_ticks)
        ax.grid(True, axis="y")
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.12)

    axes[0].set_ylabel("Identified-neuron records (stacked count)")
    handles, labels = collect_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(max(1, len(labels)), 7),
        )
    fig.suptitle(title, y=1.10, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, out_base, formats, dpi)


def overlap_rows(run: RunData) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dialect_a in run.dialects:
        set_a = run.sets_by_dialect.get(dialect_a, set())
        for dialect_b in run.dialects:
            set_b = run.sets_by_dialect.get(dialect_b, set())
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccard = intersection / union if union else 0.0
            rows.append(
                {
                    "model": run.label,
                    "dialect_a": dialect_a,
                    "dialect_b": dialect_b,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": jaccard,
                }
            )
    return rows


def plot_overlap_heatmaps(
    runs: Sequence[RunData],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    title: str,
) -> List[Path]:
    max_dialects = max(len(run.dialects) for run in runs)
    fig_width = max(6.8, (0.62 * max_dialects + 0.9) * len(runs) + 0.55)
    fig_height = max(4.2, 0.58 * max_dialects + 1.45)
    fig, axs = plt.subplots(
        1,
        len(runs),
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axs[0])
    image = None
    cmap = plt.get_cmap("viridis")

    for ax, run in zip(axes, runs):
        rows = overlap_rows(run)
        index = {(row["dialect_a"], row["dialect_b"]): row for row in rows}
        matrix: List[List[float]] = []
        for dialect_a in run.dialects:
            row_values: List[float] = []
            for dialect_b in run.dialects:
                row_values.append(float(index[(dialect_a, dialect_b)]["jaccard"]))
            matrix.append(row_values)

        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="equal")
        ax.set_title(run.label)
        ax.set_xlabel("Dialect")
        ax.set_ylabel("Dialect")
        ax.set_xticks(range(len(run.dialects)), run.dialects, rotation=45, ha="right")
        ax.set_yticks(range(len(run.dialects)), run.dialects)
        ax.tick_params(length=0)

        for i, dialect_a in enumerate(run.dialects):
            for j, dialect_b in enumerate(run.dialects):
                row = index[(dialect_a, dialect_b)]
                jaccard = float(row["jaccard"])
                intersection = int(row["intersection"])
                red, green, blue, _ = cmap(jaccard)
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                text_color = "white" if luminance < 0.48 else "#111111"
                ax.text(
                    j,
                    i,
                    f"{jaccard:.2f}\nn={intersection}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8.6,
                    linespacing=1.08,
                )

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.025)
        cbar.set_label("Jaccard overlap")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return save_figure(fig, out_base, formats, dpi)


def write_layer_counts_csv(path: Path, runs: Sequence[RunData]) -> None:
    dialect_columns: List[str] = []
    seen = set()
    for run in runs:
        for dialect in run.dialects:
            if dialect not in seen:
                seen.add(dialect)
                dialect_columns.append(dialect)

    fieldnames = ["model", "layer", "total_unique", "total_records"] + [
        f"count_{dialect}" for dialect in dialect_columns
    ]
    rows: List[Dict[str, object]] = []
    for run in runs:
        for layer in range(run.num_layers):
            row: Dict[str, object] = {
                "model": run.label,
                "layer": layer,
                "total_unique": run.layer_unique_counts.get(layer, 0),
                "total_records": run.layer_record_counts.get(layer, 0),
            }
            for dialect in dialect_columns:
                row[f"count_{dialect}"] = run.layer_counts_by_dialect.get(dialect, {}).get(layer, 0)
            rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_overlap_csv(path: Path, runs: Sequence[RunData]) -> None:
    fieldnames = ["model", "dialect_a", "dialect_b", "intersection", "union", "jaccard"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for row in overlap_rows(run):
                clean_row = dict(row)
                clean_row["jaccard"] = f"{float(row['jaccard']):.10f}"
                writer.writerow(clean_row)


def log_run_summary(runs: Sequence[RunData]) -> None:
    for run in runs:
        total_records = len(run.records)
        total_unique = len({(record.layer, record.neuron) for record in run.records})
        best_layer = max(run.layer_unique_counts, key=lambda layer: run.layer_unique_counts[layer])
        print(
            f"{run.label}: {total_records} records, {total_unique} unique neurons, "
            f"peak layer {best_layer} ({run.layer_unique_counts[best_layer]} unique)",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)
    configure_matplotlib(args.dpi)

    if args.run:
        run_specs = [parse_run_spec(value) for value in args.run]
    else:
        run_specs = [(label, path.resolve()) for label, path in DEFAULT_RUNS]

    runs = [load_run(label, input_dir) for label, input_dir in run_specs]
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_run_summary(runs)

    trend_base = out_dir / f"{args.prefix}_layer_trend"
    bar_base = out_dir / f"{args.prefix}_layer_bar"
    overlap_base = out_dir / f"{args.prefix}_overlap_jaccard"
    written = []
    written.extend(
        plot_layer_trend(
            runs=runs,
            out_base=trend_base,
            formats=formats,
            dpi=args.dpi,
            title=args.trend_title,
            show_total=not args.no_total,
            max_x_ticks=args.max_x_ticks,
        )
    )
    written.extend(
        plot_layer_bars(
            runs=runs,
            out_base=bar_base,
            formats=formats,
            dpi=args.dpi,
            title=args.bar_title,
            max_x_ticks=args.max_x_ticks,
        )
    )
    written.extend(
        plot_overlap_heatmaps(
            runs=runs,
            out_base=overlap_base,
            formats=formats,
            dpi=args.dpi,
            title=args.overlap_title,
        )
    )

    layer_table = out_dir / f"{args.prefix}_layer_counts.csv"
    overlap_table = out_dir / f"{args.prefix}_overlap_jaccard.csv"
    write_layer_counts_csv(layer_table, runs)
    write_overlap_csv(overlap_table, runs)

    print("Wrote figures:", file=sys.stderr)
    for path in written:
        print(f"  {path}", file=sys.stderr)
    print("Wrote tables:", file=sys.stderr)
    print(f"  {layer_table}", file=sys.stderr)
    print(f"  {overlap_table}", file=sys.stderr)


if __name__ == "__main__":
    main()
