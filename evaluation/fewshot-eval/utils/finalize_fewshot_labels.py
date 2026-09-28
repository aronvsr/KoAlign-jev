"""
Finalize few-shot TSV files by appending a normalized class label to `inputs`.

This script keeps original `inputs` text as-is, reads class values from
`targets` (<class>...</class>), maps them to task-specific labels, and appends
the mapped label to `inputs`.

Mappings:
    freeform: 1 -> 좋음, 0 -> 중립, -1 -> 나쁨
    yesno:    1 -> 동의, -1 -> 동의하지 않음

Usage:
    uv run ./utils/finalize_fewshot_labels.py \
      --task freeform \
      --input ./exemplars_rs_seed=42/intermediate/train.freeform.fewshot_30.tsv \
      --output ./exemplars_rs_seed=42/train.freeform.fewshot_30.tsv
"""

import argparse
import csv
import re
from pathlib import Path


CLASS_PATTERN = re.compile(r"<분류>\s*(-?\d+)\s*</분류>")
MORAL_SENTENCE_PREFIX_PATTERN = re.compile(r"^\s*\[도덕_문장\]:\s*")

LABEL_MAP = {
    "freeform": {
        -1: "나쁨",
        0: "중립",
        1: "좋음",
    },
    "yesno": {
        -1: "동의하지 않음",
        1: "동의",
    },
}


def extract_class_value(target_text: str) -> int:
    match = CLASS_PATTERN.search(target_text or "")
    if not match:
        raise ValueError(f"Failed to find <분류>...</분류> in targets: {target_text}")
    return int(match.group(1))


def strip_moral_sentence_prefix(input_text: str) -> str:
    """Remove a leading '[도덕_문장]: ' marker from inputs if present."""
    return MORAL_SENTENCE_PREFIX_PATTERN.sub("", input_text, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append class label (derived from targets) to inputs for few-shot final TSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["freeform", "yesno"],
        help="Dataset type used for class-to-label mapping",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Input intermediate TSV path",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output final TSV path",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise SystemExit("TSV header not found.")
    if "inputs" not in fieldnames or "targets" not in fieldnames:
        raise SystemExit(f"Required columns are missing. Found columns: {fieldnames}")

    label_map = LABEL_MAP[args.task]
    transformed_rows: list[dict] = []
    class_dist: dict[int, int] = {}

    for idx, row in enumerate(rows, start=1):
        inputs = str(row.get("inputs") or "")
        inputs = strip_moral_sentence_prefix(inputs)
        targets = str(row.get("targets") or "")
        cls = extract_class_value(targets)

        if cls not in label_map:
            raise SystemExit(
                f"Row {idx}: class '{cls}' is not valid for task='{args.task}'. "
                f"Supported classes: {sorted(label_map.keys())}"
            )

        mapped_label = label_map[cls]
        row["inputs"] = f"{inputs}"
        row["targets"] = f"{mapped_label}"
        transformed_rows.append(row)
        class_dist[cls] = class_dist.get(cls, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(transformed_rows)

    sorted_dist = dict(sorted(class_dist.items(), key=lambda x: x[0]))
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Task  : {args.task}")
    print(f"Rows  : {len(transformed_rows)}")
    print(f"Class distribution: {sorted_dist}")


if __name__ == "__main__":
    main()