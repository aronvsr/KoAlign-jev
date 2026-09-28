"""
KoCSNB CSV → Delphi freeform TSV convert scripts

Usage:
    python format_freeform.py --input csnb_freeform_train.csv --output train.tsv \
        --input-column "S+A,S+C+A,Q(S+A)" --class-method sum

    # Merge multiple input files
    python format_freeform.py -i file1.csv file2.csv -o merged.tsv \
        -c "S+A,Q(S+A),Q(A)" -m majority

Output format (TSV):
    inputs\ttargets
    rot\t<분류>N</분류> <판단>judgment</판단>
"""

import argparse
import ast
import csv
import random
import sys
import re
from collections import Counter
from pathlib import Path


# default: action,S+A,Q(S+A),Q(A)
INPUT_COLUMNS = ["action", "S+A", "Q(S+A)", "Q(A)"]

# default: majority
CLASS_METHODS = ["sum", "absence", "majority"]


def parse_social_judgment(raw: str) -> list[int]:
    """parses social-judgment string into a list of integers."""
    try:
        values = ast.literal_eval(raw)
        if isinstance(values, list):
            return [int(v) for v in values]
    except (ValueError, SyntaxError):
        pass
    raise ValueError(f"Failed to parse social-judgment string: {raw!r}")


def classify_majority(values: list[int]) -> int:
    """Majority-based classification. Returns the most common sign among positive/negative/zero."""
    signs = []
    for v in values:
        if v > 0:
            signs.append(1)
        elif v < 0:
            signs.append(-1)
        else:
            signs.append(0)
    counter = Counter(signs)
    # If there is a tie for the most common sign, return 0 (neutral)
    most_common = counter.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return 0
    return most_common[0][0]


CLASSIFIERS = {
    "majority": classify_majority,
}


def remove_emoji(text: str) -> str:
    """Removes emojis (and some special symbols) from a string."""
    if text is None:
        return text
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002B00-\U00002BFF"  # arrows etc.
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", str(text))


def format_row(input_text: str, class_val: int, judgment: str) -> tuple[str, str]:
    """Formats a single row into freeform format."""
    inputs = f"[도덕_문장]: {input_text}"
    # inputs = f"{input_text}"
    targets = f"<분류>{class_val}</분류> <판단>{judgment}</판단>"
    return inputs, targets


def main():
    parser = argparse.ArgumentParser(
        description="KoCSNB CSV → Delphi freeform TSV 변환"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        nargs="+",
        type=Path,
        help="Input CSV file path(s) (multiple allowed with spaces)",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output TSV file path"
    )
    parser.add_argument(
        "--input-column",
        "-c",
        required=True,
        type=str,
        help=f"Input column(s) to use for inputs (comma-separated, e.g., 'S+A,S+C+A,Q(S+A)'). Choices: {', '.join(INPUT_COLUMNS)}",
    )
    parser.add_argument(
        "--class-method",
        "-m",
        required=True,
        choices=CLASS_METHODS,
        help="Class classification method (sum: total, absence: absence, majority: majority vote)",
    )
    parser.add_argument(
        "--random-input-column",
        action="store_true",    
        help=(
            "Each row will randomly select one input column from the specified columns. "
            "The order of the columns will be shuffled to ensure uniform distribution (--seed can fix the randomness)."
        ),
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help=(
            "Random seed for --random-input-column (default: 42)"
        ),
    )
    args = parser.parse_args()

    for input_file in args.input:
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            sys.exit(1)

    # 입력 컬럼 파싱 및 검증
    input_columns = [col.strip() for col in args.input_column.split(",")]
    invalid_cols = [col for col in input_columns if col not in INPUT_COLUMNS]
    if invalid_cols:
        print(
            f"Error: Invalid input column(s): {invalid_cols}. Choices: {INPUT_COLUMNS}",
            file=sys.stderr,
        )
        sys.exit(1)

    classifier = CLASSIFIERS.get(
        args.class_method
    )
    use_majority_column = args.class_method == "majority"
    rng = random.Random(args.seed)

    rows_written = 0
    errors = 0

    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(
            fout, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        writer.writerow(["inputs", "targets"])

        for input_file in args.input:
            print(f"Processing: {input_file}")

            with open(input_file, "r", encoding="utf-8-sig") as fin:
                reader = csv.DictReader(fin)

                # 필수 컬럼 확인
                required = set(input_columns) | {"judgment"}
                if use_majority_column:
                    required.add("majority-social-judgment")
                else:
                    required.add("social-judgment")
                if reader.fieldnames is None:
                    print(
                        f"Error: CSV header not found: {input_file}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                # 파일에 존재하는 input_columns만 사용 (파일마다 컬럼이 다를 수 있음)
                available_cols = [
                    col for col in input_columns if col in reader.fieldnames
                ]
                missing_required = (required - set(input_columns)) - set(
                    reader.fieldnames
                )
                if missing_required:
                    print(
                        f"Error: Missing required column(s) ({input_file}): {missing_required}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                if not available_cols:
                    print(
                        f"Warning: No available input columns in {input_file}. Skipping.",
                        file=sys.stderr,
                    )
                    continue

                skipped_cols = set(input_columns) - set(available_cols)
                if skipped_cols:
                    print(f"  Note: Skipping columns not in {input_file}: {skipped_cols}")

                all_rows = list(reader)
                file_written = 0

                def _process_row(row: dict, col: str) -> bool:
                    nonlocal errors, rows_written, file_written
                    input_raw = row.get(col, "")
                    judgment_raw = row.get("judgment", "")
                    input_text = remove_emoji(input_raw.strip())
                    judgment = remove_emoji(judgment_raw.strip())
                    if not input_text or not judgment:
                        errors += 1
                        return False
                    if use_majority_column:
                        try:
                            class_val = int(row["majority-social-judgment"].strip())
                        except (ValueError, KeyError):
                            errors += 1
                            return False
                    else:
                        social_judgment_raw = row["social-judgment"].strip()
                        try:
                            social_values = parse_social_judgment(social_judgment_raw)
                        except ValueError:
                            errors += 1
                            return False
                        class_val = classifier(social_values)
                    inp, tgt = format_row(input_text, class_val, judgment)
                    writer.writerow([inp, tgt])
                    rows_written += 1
                    file_written += 1
                    return True

                if args.random_input_column:
                    cycle = available_cols[:]
                    rng.shuffle(cycle)
                    col_iter = (cycle[i % len(cycle)] for i in range(len(all_rows)))
                    for row, col in zip(all_rows, col_iter):
                        _process_row(row, col)
                    print(
                        f"  → {file_written}row transformed (random input column, seed={args.seed})"
                    )
                else:
                    for col in available_cols:
                        for row in all_rows:
                            _process_row(row, col)
                    print(f" → {file_written}row transformed (columns: {available_cols})")


    print(f"\nComplete: {rows_written} rows transformed, {errors} errors/skipped")

    print(f"Input files: {[str(p) for p in args.input]}")
    print(f"input columns: {input_columns}")

    print(f"Output file: {args.output}")

    class_counts: Counter = Counter()
    with open(args.output, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            targets = row["targets"]
            start = targets.index("<분류>") + len("<분류>")
            end = targets.index("</분류>")
            class_counts[int(targets[start:end])] += 1

    print(f"클래스 분포: {dict(sorted(class_counts.items()))}")


if __name__ == "__main__":
    main()
