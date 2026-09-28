# csnb-translated

Translated-corpus training condition. 3 batches. Train split only: freeform and yesno, each xlsx. 

## Format

Built from an English social-norm "Rule of Thumb" (RoT) dataset, translated to
Korean.

- **freeform**: `inputs` = `[도덕_문장]: {situation/question}`; `targets` =
  `<분류>{class}</분류> <판단>{judgment}</판단>`, class in {1, 0, -1}.
- **yesno**: each RoT produces two rows — the original ("positive") RoT and a negated
  ("negative") version. `inputs` = `[도덕_문장]: {judgment sentence}`; `targets` =
  `<분류>{1 or -1}</분류> <판단>{그렇다/아니다}, {judgment of the positive rot}</판단>`.

Row count and order match the English source. Metadata columns present in the English
source (situation_id, rot_id, src_amj, etc.) are not present here — only
`inputs`/`targets`.
