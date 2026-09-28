# jev-zeroshot

Zero-shot baseline with TypeSafe Jev (hosted API, typed/probabilistic output).

- Data: `dataset/koalign/valid.{freeform,yesno}.tsv`
- Labels: freeform `1/0/-1 -> 좋음/중립/나쁨`; yesno `1/-1 -> 동의/동의하지 않음`
- Prompt: system text from `prompts/260722_v0_1.py`, user text `상황: {situation}`; label set passed as enum.
- Prediction: argmax label. No parse failures possible; API errors reported separately
  (`acc_all` counts them wrong, `acc_answered` excludes them).
- Not directly comparable to the HF-checkpoint zero-shot protocol (no chat template,
  no greedy decoding, closed/updatable model). Record model version and run date.

```
export TYPESAFE_API_KEY=...
python evaluation/jev-zeroshot/run_jev_zeroshot.py --mode yesno --dry-run --limit 20   # pipeline check
python evaluation/jev-zeroshot/run_jev_zeroshot.py --mode yesno
python evaluation/jev-zeroshot/run_jev_zeroshot.py --mode freeform
```

Status: `call_jev()` is a stub until the API request/response format is filled in.
