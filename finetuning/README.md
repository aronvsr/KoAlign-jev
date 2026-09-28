# training SFT

Train 3 seeds (10, 21, 42) per model with those params

## Optimal params (best composite score: yesno_C2 + freeform_C3 + freeform_C2)

| Model | learning_rate | batch_size | weight_decay |
|---|---|---|---|
| EXAONE-4.0-1.2B | 2e-6 | 32 | 0.1 |
| HyperCLOVAX-SEED-Text-Instruct-0.5B | - | - | - |
| Llama-3.2-1B-Instruct | - | - | - |
| OLMo-2-0425-1B-Instruct | - | - | - |
| Midm-2.0-Mini-Instruct | - | - | - |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | - | - | - |
| Qwen3.5-2B | - | - | - |
| Llama-3.2-3B-Instruct | - | - | - |
| OLMo-2-0425-7B-Instruct | - | - | - |
| Gemma-4-E2B | - | - | - |

Fixed for all: cosine schedule, `adamw_torch_fused`, 5% warmup ratio, fp32
master weights + bf16 autocast compute, gradient checkpointing on.

Training protocol: **patient early stopping**, not a fixed epoch count. Every
5% of an epoch (`eval_steps = round(0.05 * steps_per_epoch)`, evaluate on the whole validation set
, score with `final_scoring_logic.py` (parse failures count as wrong), composite = yesno_C2 + freeform_C3 + freeform_C2.
Save the checkpoint on every new best composite. Stop only after `patience=3`
consecutive non-improving checks (not on the first dip — the subset signal is
noisy).

## Prompt / target format

Different prompt compared to zero-shot, we use (`prompts/260915_v0_2.py`).
Training target is the `<분류> 동의하지 않음 </분류>,<판단> 아니다, 그러면 안 된다. </판단>` format.

