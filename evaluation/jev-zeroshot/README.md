# jev-zeroshot

Zero-shot baseline with TypeSafe Jev (hosted API, typed/probabilistic output).

- Data: `dataset/koalign/valid.{freeform,yesno}.tsv`
- Labels: freeform `1/0/-1 -> 좋음/중립/나쁨`; yesno `1/-1 -> 동의/동의하지 않음`
- Request (typesafe-sdk `system_one`): `state={"situation": ...}`, one `Choice` question with
  the system text from `prompts/260722_v0_1.py` as `instructions` and the labels as `criteria`.
  Model: `--model` (default `jev-latest`); the served model name is saved per row.
- Prediction: argmax label. No parse failures possible; API errors reported separately
  (`acc_all` counts them wrong, `acc_answered` excludes them).
- Not directly comparable to the HF-checkpoint zero-shot protocol (no chat template,
  no greedy decoding, closed/updatable model). Record model version and run date.

```
pip install typesafe-sdk
export TYPESAFE_API_KEY=...
python evaluation/jev-zeroshot/run_jev_zeroshot.py --mode yesno --dry-run --limit 20   # pipeline check
python evaluation/jev-zeroshot/run_jev_zeroshot.py --mode yesno
python evaluation/jev-zeroshot/run_jev_zeroshot.py --mode freeform
```

Status: wired to typesafe-sdk 0.7.2 and tested against a mocked response; not yet run against
the live API (`api.typesafe.ai` is blocked from the cloud session that wrote this).
