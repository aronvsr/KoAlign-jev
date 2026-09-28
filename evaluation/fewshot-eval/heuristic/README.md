# Heuristic Exemplars (seed=42)

- Class labels are distributed as uniformly as possible.
- Same seed keeps exemplar ordering consistent across shot counts.

`./intermediate/train.*.sampling.{a,b,c}.tsv` are researcher-curated candidate pools.

## Updated process

Intermediate files:

- `./intermediate/train.{freeform,yesno}.fewshot_{3,10,30}.tsv`

Final files:

- `./train.{freeform,yesno}.fewshot_{3,10,30}.tsv`

### 1) Balanced extraction from heuristic pools (intermediate TSV)

```bash
uv run ./utils/extract_fewshot_heuristic.py --input ./heuristic/intermediate/train.freeform.sampling.a.tsv ./heuristic/intermediate/train.freeform.sampling.b.tsv ./heuristic/intermediate/train.freeform.sampling.c.tsv --n {3,10,30} --seed 42 --output ./heuristic/intermediate/train.freeform.fewshot_{3,10,30}.tsv
uv run ./utils/extract_fewshot_heuristic.py --input ./heuristic/intermediate/train.yesno.sampling.a.tsv ./heuristic/intermediate/train.yesno.sampling.b.tsv ./heuristic/intermediate/train.yesno.sampling.c.tsv --n {3,10,30} --seed 42 --output ./heuristic/intermediate/train.yesno.fewshot_{3,10,30}.tsv
```

### 2) Finalize: append mapped class label to `inputs`

```bash
uv run ./utils/finalize_fewshot_labels.py --task freeform --input ./heuristic/intermediate/train.freeform.fewshot_{3,10,30}.tsv --output ./heuristic/train.freeform.fewshot_{3,10,30}.tsv
uv run ./utils/finalize_fewshot_labels.py --task yesno --input ./heuristic/intermediate/train.yesno.fewshot_{3,10,30}.tsv --output ./heuristic/train.yesno.fewshot_{3,10,30}.tsv
```

Label mapping used in finalization:

- freeform: `1 -> 좋음`, `0 -> 중립`, `-1 -> 나쁨`
- yesno: `1 -> 동의`, `-1 -> 동의하지 않음`

