# Exemplars with Random Sampling (seed=42)

- Class labels are distributed as uniformly as possible.
- Data sources (`crd`, `kbsn`) are distributed as uniformly as possible.
- Same seed keeps exemplar ordering consistent across shot counts.

## Updated process

Intermediate files:

- `./intermediate/train.{freeform,yesno}.fewshot_{3,10,30}.csv`
- `./intermediate/train.{freeform,yesno}.fewshot_{3,10,30}.tsv`

Final files:

- `./train.{freeform,yesno}.fewshot_{3,10,30}.tsv`

### 1) Sample from KoAlign CSV

```bash
uv run ./utils/extract_fewshot.py --input ../../dataset/koalign/train.freeform.koalign.csv --n {3,10,30} --seed 42 --output ./exemplars_rs_seed=42/intermediate/train.freeform.fewshot_{3,10,30}.csv
uv run ./utils/extract_fewshot.py --input ../../dataset/koalign/train.yesno.koalign.csv --n {3,10,30} --seed 42 --output ./exemplars_rs_seed=42/intermediate/train.yesno.fewshot_{3,10,30}.csv
```

### 2) Convert to Delphi TSV (intermediate)

```bash
uv run ./utils/format_freeform.py --input ./exemplars_rs_seed=42/intermediate/train.freeform.fewshot_{3,10,30}.csv --input-column "action,S+A,Q(S+A),Q(A)" -m majority --output ./exemplars_rs_seed=42/intermediate/train.freeform.fewshot_{3,10,30}.tsv --random-input-column
uv run ./utils/format_yesno.py --input ./exemplars_rs_seed=42/intermediate/train.yesno.fewshot_{3,10,30}.csv --output ./exemplars_rs_seed=42/intermediate/train.yesno.fewshot_{3,10,30}.tsv
```

### 3) Finalize: append mapped class label to `inputs`

```bash
uv run ./utils/finalize_fewshot_labels.py --task freeform --input ./exemplars_rs_seed=42/intermediate/train.freeform.fewshot_{3,10,30}.tsv --output ./exemplars_rs_seed=42/train.freeform.fewshot_{3,10,30}.tsv
uv run ./utils/finalize_fewshot_labels.py --task yesno --input ./exemplars_rs_seed=42/intermediate/train.yesno.fewshot_{3,10,30}.tsv --output ./exemplars_rs_seed=42/train.yesno.fewshot_{3,10,30}.tsv
```

Label mapping used in finalization:

- freeform: `1 -> 좋음`, `0 -> 중립`, `-1 -> 나쁨`
- yesno: `1 -> 동의`, `-1 -> 동의하지 않음`

