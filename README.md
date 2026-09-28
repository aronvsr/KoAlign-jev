# KoAlign

## Repository layout (저장소 구조)

```
KoAlign/
  dataset/
    koalign/              native training/eval corpus: train/valid/test, tsv + rich CSV
    csnb-translated/       translated-corpus training condition: train tsv/jsonl/xlsx
  prompts/                prompt-construction config
  evaluation/
    zeroshot-eval/         zero-shot evaluation conventions
    fewshot-eval/          few-shot evaluation conventions (3/10/30-shot) + exemplars
    koalign-eval/          evaluation results for models fine-tuned on dataset/koalign
    csnb-translated-eval/  evaluation results for models fine-tuned on dataset/csnb-translated
  finetuning/              fine-tuning conventions
  models/                  fine-tuned model checkpoints/outputs
  README.md
```

Code in this repository: `prompts/260722_v0_1.py` + `prompts/__init__.py` (prompt
config), `evaluation/fewshot-eval/extract_fewshot.py` (exemplar sampler),
`src/training/config.py` (`260722_v0_1.py` imports
`OutputMode`, `SPECIAL_TOKENS`). No other code exists in this repository.

## Model justification (모델 선정 이유)

Selection criteria: a size range from ~0.5B to ~32B; state-of-the-art or near
state-of-the-art checkpoints within each family at the time of selection; a mix of
Korean-focused models (EXAONE-4.0, HyperCLOVAX-SEED-Text-Instruct, Midm-2.0) and
global models (Qwen3.5, Gemma-4, Llama, OLMo-2), to compare Korean-language
performance across models built for Korean specifically versus general-purpose
multilingual models.

## Model lineup (모델 목록)

| Family | Size | HF checkpoint | Fine-tuned (<=7B, 파인튜닝 대상) | Precision |
|---|---|---|---|---|
| EXAONE-4.0 | 1.2B | `LGAI-EXAONE/EXAONE-4.0-1.2B` | yes | bf16 |
| EXAONE-4.0 | 32B | `LGAI-EXAONE/EXAONE-4.0-32B` | no | bf16 |
| HyperCLOVAX-SEED-Text-Instruct | 0.5B | `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B` | yes | bf16 |
| HyperCLOVAX-SEED-Text-Instruct | 1.5B | `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | yes | bf16 |
| Midm-2.0 | Mini-Instruct (2B) | `K-intelligence/Midm-2.0-Mini-Instruct` | yes | bf16 |
| Midm-2.0 | Base-Instruct (12B) | `K-intelligence/Midm-2.0-Base-Instruct` | no | bf16 |
| Qwen3.5 | 2B | `Qwen/Qwen3.5-2B` | yes | bf16 |
| Qwen3.5 | 9B | `Qwen/Qwen3.5-9B` | no | bf16 |
| Qwen3.5 | 27B | `Qwen/Qwen3.5-27B` | no | bf16 |
| Gemma-4 | E2B | `google/gemma-4-E2B-it` | TBD | bf16 |
| Gemma-4 | 12B | `google/gemma-4-12B-it` | no | bf16 |
| Gemma-4 | 31B | `google/gemma-4-31B-it` | no | bf16 |
| Llama | 3.2-1B(-Instruct) | `meta-llama/Llama-3.2-1B[-Instruct]` | yes | bf16 |
| Llama | 3.2-3B(-Instruct) | `meta-llama/Llama-3.2-3B[-Instruct]` | yes | bf16 |
| Llama | 3.1-8B(-Instruct) | `meta-llama/Llama-3.1-8B-Instruct` | no | bf16 |
| OLMo-2 | 1B (0425-Instruct) | `allenai/OLMo-2-0425-1B-Instruct` | yes | bf16 |
| OLMo-2 | 7B | `allenai/OLMo-2-1124-7B-Instruct` | yes | bf16 |
| OLMo-2 | 32B | `allenai/OLMo-2-0325-32B-Instruct` | no | bf16 |


## Evaluation protocols (평가 방식)

Add your results to [this Google Sheet](https://docs.google.com/spreadsheets/d/1TM1yoPX4B5_1OME9uPN1nU1p_R8NRNEFB4hA2YuZnLE/edit?usp=sharing).

All three protocols evaluate against the same validation set:
`dataset/koalign/valid.{freeform,yesno}.tsv`, byte-identical to the implicit
evaluation target for `dataset/csnb-translated/` (which has no valid split of its
own).

### Zero-shot (제로샷)
One deterministic run per model. Decoding: greedy (`do_sample=False`). No seeds, no
averaging. Prompts: `prompts/260722_v0_1.py`.

### Few-shot (퓨샷)
Shot counts: 3, 10, 30. Three exemplar draws per shot-count, each evaluated once
(greedy). Exemplars for draw 1: `evaluation/fewshot-eval/exemplars/train.{freeform,yesno}.fewshot_{3,10,30}.tsv`
(`inputs`/`targets` TSV format). Draws 2 and 3 do not exist.

`evaluation/fewshot-eval/extract_fewshot.py` produces stratified samples (judgment
-1/0/1 x source crd/kbsn, seeded) from `dataset/koalign/train.*.koalign.csv`. Its
output columns (`situation`, `action`, `judgment`, `rot`, ...) do not match the
`inputs`/`targets` TSV format used by draw 1; no conversion between the two exists.
Draw 1 was produced by a different process and is not methodologically equivalent to
draws made with `extract_fewshot.py`.

Exemplar answers must render as the categorical label (좋음/나쁨/중립 or
동의/동의하지 않음, matching the system prompt's instructed output), not as free
judgment text — rendering free text as the exemplar answer instead of the categorical
label caused parse failure rates of 0-12% (yesno) and 0.1-39% (freeform) against a
95-100% baseline in prior testing.

## Precision (정밀도)

**bf16** for model weights, for every model in the lineup, fine-tuned or
evaluation-only. All checkpoints are released in bf16 upstream; running inference or
training at fp32 does not add precision the models were trained with, and fp16's
narrower dynamic range is less stable for transformers than bf16. bf16 also roughly
halves memory versus fp32 — relevant given fp32 weights for a 32B model would require
>120GB, exceeding Aron's hardware (4x RTX A5000, 24GB each, 96GB total).

## Conventions (공통 설정)

| Variable | Value |
|---|---|
| `temperature` | 0 (`do_sample=False`) |
| `top_p` | 1 |
| `top_k` | disabled |
| Precision | bf16 weights (fine-tuning: fp32 optimizer states) |
| Few-shot exemplar draws | 3 per shot-count |
| Few-shot shot counts | 3, 10, 30 |
| Fine-tuning seeds | 42, 21, 10, 7, 2 |
| Fine-tuning hyperparameters | LR 1e-5, 4 epochs, warmup 256, cosine schedule |
| Eval metrics | C2, C3, T(A), parse_success_rate |
| Std convention | ddof=1 (sample std) |
| Prompt construction | `prompts/260722_v0_1.py` |

## Open items (미결 사항)

1. Few-shot heuristic problems. (Kyungkyu working on it).
