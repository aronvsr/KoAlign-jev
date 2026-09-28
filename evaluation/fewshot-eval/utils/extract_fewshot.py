"""
Few-shot exemapler Exctraction Scripts

It extracts N few-shot examples from the train CSV file with uniform sampling.


Conditions for uniform sampling:
    - majority-social-judgment: -1, 0, 1 are uniformly distributed
    - source (normalized): crd, kbsn are uniformly distributed
    → Sample as uniformly as possible from 6 buckets (judgment × source)
    - source: crd=ceil(N/2), kbsn=floor(N/2)
    - Each source's judgment (-1/0/1): divide the source quota by 3

Usage:
    uv run extract_fewshot.py --input ./output/26-04-29-kocsnb-basic/train.freeform.26-04-29-kocsnb-basic.csv --n 30
    uv run extract_fewshot.py --input ./output/26-04-29-kocsnb-basic/train.freeform.26-04-29-kocsnb-basic.csv --n 30 --seed 42
    uv run extract_fewshot.py --input ./output/26-04-29-kocsnb-basic/train.freeform.26-04-29-kocsnb-basic.csv --n 30 --output ./fewshot_examples.csv
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


# ─── Source Regular ─────────────────────────────────────────────────────

def normalize_source(source: str) -> str:
    """
    crd-second / crd_2nd → "crd"
    kbsn-second / kbsn_2nd → "kbsn"
    Etc → keep as lowercase
    """
    s = source.lower().strip()
    if s.startswith("crd"):
        return "crd"
    if s.startswith("kbsn"):
        return "kbsn"
    return s


# ─── 메인 ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Few-shot exeampler extraction from train CSV file with uniform sampling",
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Input train CSV file path",
    )
    parser.add_argument(
        "--n", "-n", type=int, default=30,
        help="Number of few-shot examples to extract (default: 30)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output file path (default: next to the input file as fewshot_N_seed.csv)",
    )
    parser.add_argument(
        "--judgment-col", type=str, default=None,
        help="Judgment column (default: auto-detect — freeform=majority-social-judgment, yesno=class_label)",
    )
    args = parser.parse_args()

    # ── CSV 로드 ──────────────────────────────────────────────────────
    print(f"Input : {args.input}")
    with open(args.input, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows")

    # ── 판단값 컬럼 결정 (자동 감지) ─────────────────────────────────
    JUDGMENT_COL_CANDIDATES = ["majority-social-judgment", "class_label"]
    if args.judgment_col:
        judgment_col = args.judgment_col
    else:
        judgment_col = next(
            (c for c in JUDGMENT_COL_CANDIDATES if c in fieldnames), None
        )
        if judgment_col is None:
            print(f"❌ Judgment column not found. Specify it directly with --judgment-col.")
            print(f"   Available columns: {fieldnames}")
            raise SystemExit(1)
    print(f"Judgment column: '{judgment_col}'")

    # ── 실제 데이터에서 judgment 값 감지 ─────────────────────────────
    SOURCES = ["crd", "kbsn"]
    JUDGMENTS = sorted(
        {r.get(judgment_col, "").strip() for r in rows if r.get(judgment_col, "").strip()},
        key=lambda x: int(x) if x.lstrip("-").isdigit() else x,
    )
    print(f"Detected judgment values: {JUDGMENTS}")

    # ── 버킷당 할당량 계산 ────────────────────────────────────────────
    # 버킷 수 = len(JUDGMENTS) × 2(source)
    # round-robin 순서: judgment와 source를 교차(interleave) 배정
    #   i번째 → jdg=JUDGMENTS[i%J], src=SOURCES[(i%J + i//J) % 2]
    # 예) yesno (J=2): (-1,crd),(1,kbsn),(-1,kbsn),(1,crd)
    #     rem=2 → -1:15, 1:15 / crd:15, kbsn:15  ← 완전 균등
    # 예) freeform (J=3): (-1,crd),(0,kbsn),(1,crd),(-1,kbsn),(0,crd),(1,kbsn)
    NUM_BUCKETS = len(JUDGMENTS) * 2
    J = len(JUDGMENTS)
    ROUND_ROBIN_ORDER = [
        (JUDGMENTS[i % J], SOURCES[(i % J + i // J) % 2])
        for i in range(NUM_BUCKETS)
    ]
    BUCKET_ORDER = [(src, jdg) for src in SOURCES for jdg in JUDGMENTS]

    base = args.n // NUM_BUCKETS
    remainder = args.n % NUM_BUCKETS
    bucket_quota: dict[tuple[str, str], int] = {key: base for key in BUCKET_ORDER}
    for jdg, src in ROUND_ROBIN_ORDER[:remainder]:
        bucket_quota[(src, jdg)] += 1

    # ── freeform: Randomly select one RoT variation ─────────────────
    # In freeform data, there may be multiple RoT variations for the same
    # (situation, action) pair, so we randomly select one per group before
    # constructing the buckets.
    is_freeform = "rot" in fieldnames
    if is_freeform:
        rot_rng = random.Random(args.seed)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            key = (r.get("situation", "").strip(), r.get("action", "").strip())
            groups[key].append(r)
        original_count = len(rows)
        rows = [rot_rng.choice(group) for group in groups.values()]
        print(f"After removing RoT variations: {original_count} rows → {len(rows)} rows ({len(groups)} groups)")

    # ── 버킷 분류 ─────────────────────────────────────────────────────
    # 버킷 키: (normalized_source, judgment)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped = 0

    for r in rows:
        norm_src = normalize_source(r.get("source", ""))
        judgment = r.get(judgment_col, "").strip()

        if norm_src not in SOURCES or judgment not in JUDGMENTS:
            skipped += 1
            continue

        buckets[(norm_src, judgment)].append(r)

    print(f"\nBucket status (total target: {args.n} rows):")
    for src in SOURCES:
        for jdg in JUDGMENTS:
            key = (src, jdg)
            quota = bucket_quota[key]
            print(f"  source={src}, judgment={jdg}: {len(buckets[key])} rows (target {quota})")
    if skipped:
        print(f"  ⚠️ Rows skipped due to unrecognized source/judgment: {skipped} rows")

    # ── 샘플링 ────────────────────────────────────────────────────────
    # 재현성 보장 규칙:
    # 1) 각 버킷 내부를 seed 기반으로 고정 셔플
    # 2) ROUND_ROBIN_ORDER 순서대로 전역 후보를 생성
    # 3) 앞에서 n개를 선택
    # 이렇게 하면 같은 seed에서 n=3 결과가 n=10, n=30 결과의 앞부분으로 항상 포함된다.
    rng = random.Random(args.seed)
    shuffled_buckets: dict[tuple[str, str], list[dict]] = {}
    for key in BUCKET_ORDER:
        pool = buckets[key][:]
        rng.shuffle(pool)
        shuffled_buckets[key] = pool

    short_buckets: list[str] = []
    for key in BUCKET_ORDER:
        src, jdg = key
        quota = bucket_quota[key]
        if len(shuffled_buckets[key]) < quota:
            short_buckets.append(f"source={src}, judgment={jdg} ({len(shuffled_buckets[key])}개 < {quota})")

    ordered_candidates: list[dict] = []
    idx_by_bucket: dict[tuple[str, str], int] = {key: 0 for key in BUCKET_ORDER}
    while True:
        progressed = False
        for jdg, src in ROUND_ROBIN_ORDER:
            key = (src, jdg)
            idx = idx_by_bucket[key]
            pool = shuffled_buckets[key]
            if idx < len(pool):
                ordered_candidates.append(pool[idx])
                idx_by_bucket[key] = idx + 1
                progressed = True
        if not progressed:
            break

    if args.n > len(ordered_candidates):
        print(
            f"⚠️ Requested n({args.n}) is larger than available valid rows({len(ordered_candidates)}). "
            "Returning all available rows."
        )

    sampled = ordered_candidates[: args.n]

    if short_buckets:
        print(f"\n⚠️ The following buckets are below the target sample size:")
        for s in short_buckets:
            print(f"   {s}")

    print(f"\nExtraction result: {len(sampled)} rows (seed={args.seed})")

    # 분포 확인 출력
    from collections import Counter
    jdg_dist = Counter(r.get(judgment_col, "").strip() for r in sampled)
    src_dist = Counter(normalize_source(r.get("source", "")) for r in sampled)
    print(f"  {judgment_col} Distribution: {dict(sorted(jdg_dist.items()))}")
    print(f"  source distribution:                   {dict(sorted(src_dist.items()))}")

    # ── 출력 ──────────────────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        stem = args.input.stem
        out_path = args.input.parent / f"fewshot_{args.n}_seed{args.seed}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sampled)

    print(f"\n✅ Saved successfully: {out_path}")


if __name__ == "__main__":
    main()
