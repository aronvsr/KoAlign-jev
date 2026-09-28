"""
KoCSNB Yes/No CSV → Delphi YesNo TSV 변환 스크립트

Supports two input formats:
  1) rot, class_label, text_label
     → 그대로 [도덕_문장]: rot → <분류>class_label</분류> <판단>text_label</판단>

  2) rot, rot_neg, rot_output, rot_neg_output (deprecated)
     → rot + rot_output → class 1
     → rot_neg + rot_neg_output → class -1

사용법:
    python format_yesno.py --input kbsn_yesno_train.csv --output train.tsv

    # 여러 입력 파일 통합
    python format_yesno.py -i file1.csv file2.csv -o merged.tsv
"""

import argparse
import csv
import sys
import re
from collections import Counter
from pathlib import Path


def detect_format(fieldnames: list[str]) -> str:
    
    has_neg = "rot_neg" in fieldnames and "rot_neg_output" in fieldnames
    has_class = "class_label" in fieldnames and "text_label" in fieldnames

    if has_neg:
        return "yesno"  # rot, rot_neg, rot_output, rot_neg_output
    elif has_class:
        return "labeled"  # rot, class_label, text_label
    else:
        return "unknown"

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

def format_row(input_text: str, class_val: int, text_label: str) -> tuple[str, str]:
    """Formats a single row into freeform format."""
    inputs = f"[도덕_문장]: {input_text}"
    targets = f"<분류>{class_val}</분류> <판단>{text_label}</판단>"
    return inputs, targets


def main():
    parser = argparse.ArgumentParser(
        description="KoCSNB Yes/No CSV → Delphi Yesno TSV conversion"
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
    args = parser.parse_args()

    for input_file in args.input:
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            sys.exit(1)

    rows_written = 0
    errors = 0
    error_details: Counter = Counter()  # 오류 유형별 카운트

    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(
            fout, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        writer.writerow(["inputs", "targets"])

        for input_file in args.input:
            print(f"Processing: {input_file}")

            with open(input_file, "r", encoding="utf-8-sig") as fin:
                reader = csv.DictReader(fin)

                if reader.fieldnames is None:
                    print(
                        f"Error: CSV header not found: {input_file}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                fmt = detect_format(reader.fieldnames)
                if fmt == "unknown":
                    print(
                        f"Error: Unsupported CSV format ({input_file}). Header: {reader.fieldnames}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                print(f"  Detected format: {fmt}")
                file_written = 0

                for i, row in enumerate(reader, start=1):
                    if fmt == "labeled":
                        rot = row["rot"].strip()
                        try:
                            class_val = int(row["class_label"].strip())
                        except (ValueError, KeyError) as e:
                            errors += 1
                            error_details["labeled: failed to parse class_label"] += 1
                            continue
                        text_label = remove_emoji(row["text_label"].strip())

                        if not rot or not text_label:
                            errors += 1
                            error_details["labeled: empty rot or text_label"] += 1
                            continue

                        inputs, targets = format_row(rot, class_val, text_label)
                        writer.writerow([inputs, targets])
                        rows_written += 1
                        file_written += 1

                    elif fmt == "yesno":
                        rot = remove_emoji(row["rot"].strip())
                        rot_neg = remove_emoji(row["rot_neg"].strip())
                        rot_output = remove_emoji(row["rot_output"].strip())
                        rot_neg_output = remove_emoji(row["rot_neg_output"].strip())

                        if rot and rot_output:
                            inputs, targets = format_row(rot, 1, rot_output)
                            writer.writerow([inputs, targets])
                            rows_written += 1
                            file_written += 1
                        else:
                            errors += 1
                            error_details["yesno: empty rot or rot_output"] += 1

                        if rot_neg and rot_neg_output:
                            inputs, targets = format_row(rot_neg, -1, rot_neg_output)
                            writer.writerow([inputs, targets])
                            rows_written += 1
                            file_written += 1
                        else:
                            errors += 1
                            error_details["yesno: empty rot_neg or rot_neg_output"] += 1

                print(f"  → {file_written} rows converted")

    print(f"\nDone: {rows_written} rows converted, {errors} errors/skipped")
    if error_details:
        print("Error details:")
        for err_type, count in error_details.most_common():
            print(f"  - {err_type}: {count} cases")
    print(f"Input files: {[str(p) for p in args.input]}")
    print(f"Output: {args.output}")

    # 클래스 분포 출력
    class_counts: Counter = Counter()
    with open(args.output, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            targets_str = row["targets"]
            start = targets_str.index("<분류>") + len("<분류>")
            end = targets_str.index("</분류>")
            class_counts[int(targets_str[start:end])] += 1

    print(f"Class distribution: {dict(sorted(class_counts.items()))}")


if __name__ == "__main__":
    main()
