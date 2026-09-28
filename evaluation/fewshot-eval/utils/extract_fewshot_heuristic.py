"""
Few-shot TSV extraction script for heuristic sampling.

This script samples N rows from a TSV file while keeping the class labels
inside <분류>...</분류> as balanced as possible.

Usage:
    uv run extract_fewshot_heuristic.py --input ./train.freeform.tsv --n 30
    uv run extract_fewshot_heuristic.py --input ./train.freeform.tsv --n 30 --seed 42
    uv run extract_fewshot_heuristic.py --input ./train.freeform.tsv --n 30 --output ./fewshot_examples.tsv
"""

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path


CLASS_PATTERN = re.compile(r"<분류>\s*(-?\d+)\s*</분류>")


def extract_class_value(target_text: str) -> str | None:
    """Extract the <분류> value from a targets string."""
    match = CLASS_PATTERN.search(target_text or "")
    if not match:
        return None
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample TSV rows with balanced <분류> distribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        help="Input TSV file path(s)",
    )
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=30,
        help="Number of rows to extract",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output TSV file path (default: fewshot_N_seed.tsv next to input)",
    )
    args = parser.parse_args()

    input_paths = args.input
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input file not found: {', '.join(str(path) for path in missing)}")

    fieldnames: list[str] = []
    rows: list[dict] = []
    for path in input_paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            current_fieldnames = list(reader.fieldnames or [])
            current_rows = list(reader)

        if not fieldnames:
            fieldnames = current_fieldnames
        elif fieldnames != current_fieldnames:
            raise SystemExit(
                f"TSV header mismatch: {input_paths[0]} {fieldnames} vs {path} {current_fieldnames}"
            )

        rows.extend(current_rows)

    if not fieldnames:
        raise SystemExit("TSV header not found.")
    if "targets" not in fieldnames:
        raise SystemExit(f"'targets' column not found. Available columns: {fieldnames}")

    class_values = sorted(
        {
            class_value
            for row in rows
            if (class_value := extract_class_value(str(row.get("targets") or "")))
        },
        key=lambda x: int(x) if x.lstrip("-").isdigit() else x,
    )
    if not class_values:
        raise SystemExit("No <분류>...</분류> values found in targets.")

    if args.n < len(class_values):
        raise SystemExit(
            f"Sample size({args.n}) is smaller than number of classes({len(class_values)})."
        )

    buckets: dict[str, list[dict]] = {cls: [] for cls in class_values}
    skipped = 0
    skipped_rows: list[tuple[str, dict]] = []
    for row in rows:
        target_text = str(row.get("targets") or "")
        class_value = extract_class_value(target_text)
        if class_value is None:
            skipped += 1
            skipped_rows.append(("missing_class", row))
            continue
        if class_value not in buckets:
            skipped += 1
            skipped_rows.append((f"unknown_class={class_value}", row))
            continue
        buckets[class_value].append(row)

    # 재현성 보장 규칙:
    # 1) 클래스별로 seed 기반 고정 셔플
    # 2) 라운드로빈으로 전역 순서 생성
    # 3) 앞에서 n개 선택
    # 이렇게 하면 같은 seed에서 n=3 결과가 n=10, n=30 결과의 앞부분으로 항상 포함된다.
    rng = random.Random(args.seed)
    shuffled_buckets: dict[str, list[dict]] = {}
    for cls in class_values:
        pool = buckets[cls][:]
        rng.shuffle(pool)
        shuffled_buckets[cls] = pool

    short_buckets: list[str] = []
    base = args.n // len(class_values)
    remainder = args.n % len(class_values)
    class_quota: dict[str, int] = {cls: base for cls in class_values}
    for cls in class_values[:remainder]:
        class_quota[cls] += 1
    for cls in class_values:
        if len(shuffled_buckets[cls]) < class_quota[cls]:
            short_buckets.append(f"class={cls} ({len(shuffled_buckets[cls])} < {class_quota[cls]})")

    ordered_candidates: list[dict] = []
    idx_by_class: dict[str, int] = {cls: 0 for cls in class_values}
    while True:
        progressed = False
        for cls in class_values:
            idx = idx_by_class[cls]
            pool = shuffled_buckets[cls]
            if idx < len(pool):
                ordered_candidates.append(pool[idx])
                idx_by_class[cls] = idx + 1
                progressed = True
        if not progressed:
            break

    if args.n > len(ordered_candidates):
        print(
            f"⚠️ Requested n({args.n}) is larger than available valid rows({len(ordered_candidates)}). "
            "Returning all available rows."
        )
    sampled = ordered_candidates[: args.n]

    out_path = args.output or input_paths[0].parent / f"fewshot_{args.n}_seed{args.seed}.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sampled)

    class_dist = Counter()
    for row in sampled:
        class_value = extract_class_value(str(row.get("targets") or ""))
        if class_value is not None:
            class_dist[class_value] += 1

    print(f"Input : {[str(path) for path in input_paths]}")
    print(f"Loaded rows: {len(rows)}")
    print(f"Detected class values: {class_values}")
    print(f"Skipped rows: {skipped}")
    if skipped_rows:
        print("Skipped row details:")
        for idx, (reason, row) in enumerate(skipped_rows, start=1):
            print(f"  [{idx}] reason={reason}")
            print(f"      inputs: {row.get('inputs', '')}")
            print(f"      targets: {row.get('targets', '')}")
    if short_buckets:
        print("Short buckets:")
        for item in short_buckets:
            print(f"  {item}")
    print(f"Class distribution: {dict(sorted(class_dist.items(), key=lambda x: int(x[0])))}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
