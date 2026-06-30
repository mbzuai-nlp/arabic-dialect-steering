#!/usr/bin/env python3
"""Create random TSV sample splits, optionally mirrored across paired files."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ID_COLUMN_CANDIDATES = ("sent_ID", "sentID", "sentID.BTEC", "id", "ID")


def parse_size(value: str) -> int:
    text = value.strip().lower()
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1000
        text = text[:-1]

    if not text or not text.isdigit():
        raise argparse.ArgumentTypeError(
            f"Invalid size '{value}'. Use an integer like 1000 or a k suffix like 1k."
        )

    size = int(text) * multiplier
    if size <= 0:
        raise argparse.ArgumentTypeError("Sample sizes must be positive.")
    return size


def size_label(size: int) -> str:
    if size % 1000 == 0:
        return f"{size // 1000}k"
    return str(size)


def detect_id_column(fieldnames: list[str], requested: str | None) -> str:
    if requested:
        if requested not in fieldnames:
            raise ValueError(
                f"Requested ID column '{requested}' was not found. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        return requested

    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate

    lower_to_name = {name.lower(): name for name in fieldnames}
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate.lower() in lower_to_name:
            return lower_to_name[candidate.lower()]

    raise ValueError(
        "Could not auto-detect the ID column. "
        f"Tried: {', '.join(ID_COLUMN_CANDIDATES)}. "
        "Pass it explicitly with --id-column."
    )


def read_tsv(path: Path, id_column: str | None) -> tuple[list[str], str, dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")

        fieldnames = reader.fieldnames
        resolved_id_column = detect_id_column(fieldnames, id_column)
        rows_by_id: dict[str, dict[str, str]] = {}
        duplicate_ids: list[str] = []

        for row_number, row in enumerate(reader, start=2):
            sample_id = row.get(resolved_id_column, "")
            if not sample_id:
                raise ValueError(f"{path}:{row_number} has an empty ID in '{resolved_id_column}'.")
            if sample_id in rows_by_id:
                duplicate_ids.append(sample_id)
                continue
            rows_by_id[sample_id] = row

    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(
            f"{path} contains duplicate IDs in '{resolved_id_column}' "
            f"(examples: {preview}). Samples must be unique."
        )

    return fieldnames, resolved_id_column, rows_by_id


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def output_path(out_dir: Path, input_path: Path, size: int) -> Path:
    return out_dir / f"{input_path.stem}.sample_{size_label(size)}{input_path.suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample TSV files by sent ID. The primary file is sampled randomly; "
            "paired files are filtered to the same sampled IDs."
        )
    )
    parser.add_argument("--primary", required=True, type=Path, help="TSV file to sample from.")
    parser.add_argument(
        "--paired",
        nargs="*",
        default=[],
        type=Path,
        help="Optional TSV files to mirror using the sampled IDs from --primary.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        required=True,
        type=parse_size,
        help="Sample sizes, e.g. --sizes 1k 2k 4k 6k 12k.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where sampled TSV files will be written.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="ID column name. If omitted, common names like sent_ID and sentID.BTEC are auto-detected.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [args.primary, *args.paired]

    primary_fields, primary_id_column, primary_rows_by_id = read_tsv(args.primary, args.id_column)
    all_ids = list(primary_rows_by_id)
    rng = random.Random(args.seed)
    rng.shuffle(all_ids)

    max_size = max(args.sizes)
    if max_size > len(all_ids):
        raise ValueError(
            f"Requested {max_size} samples, but {args.primary} only has "
            f"{len(all_ids)} unique samples."
        )

    paired_data: dict[Path, tuple[list[str], dict[str, dict[str, str]]]] = {}
    for paired_path in args.paired:
        paired_fields, paired_id_column, paired_rows_by_id = read_tsv(paired_path, args.id_column)
        if paired_id_column != primary_id_column:
            print(
                f"Note: {paired_path} uses ID column '{paired_id_column}' "
                f"while primary uses '{primary_id_column}'. Matching by ID values."
            )
        paired_data[paired_path] = (paired_fields, paired_rows_by_id)

    print(f"Primary: {args.primary} ({len(all_ids)} unique samples)")
    print(f"ID column: {primary_id_column}")
    print(f"Output directory: {args.out_dir}")

    for size in sorted(set(args.sizes)):
        selected_ids = all_ids[:size]

        primary_rows = [primary_rows_by_id[sample_id] for sample_id in selected_ids]
        primary_out = output_path(args.out_dir, args.primary, size)
        write_tsv(primary_out, primary_fields, primary_rows)
        print(f"Wrote {len(primary_rows):>6} rows -> {primary_out}")

        for paired_path, (paired_fields, paired_rows_by_id) in paired_data.items():
            missing_ids = [sample_id for sample_id in selected_ids if sample_id not in paired_rows_by_id]
            if missing_ids:
                preview = ", ".join(missing_ids[:5])
                raise ValueError(
                    f"{paired_path} is missing {len(missing_ids)} sampled IDs "
                    f"for size {size_label(size)} (examples: {preview})."
                )

            paired_rows = [paired_rows_by_id[sample_id] for sample_id in selected_ids]
            paired_out = output_path(args.out_dir, paired_path, size)
            write_tsv(paired_out, paired_fields, paired_rows)
            print(f"Wrote {len(paired_rows):>6} rows -> {paired_out}")


if __name__ == "__main__":
    main()
