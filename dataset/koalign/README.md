# koalign

Native training/eval corpus. train/valid/test splits, each as plain 2-column
(`inputs`/`targets`) TSV and rich CSV (source, situation, action, judgment,
majority-social-judgment, rot, and other annotation fields).

`valid.freeform.tsv` / `valid.yesno.tsv` are the fixed validation set for all
evaluation protocols in this repository, including the `csnb-translated` condition,
which has no valid split of its own.

`train.*.koalign.csv` (rich format) is read by
`evaluation/fewshot-eval/extract_fewshot.py`.




## TODO

- Add Code and documentation for consturct dataset