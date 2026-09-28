# prompts/

- `__init__.py` — `PromptConfig` base class, `load_prompt_config(name)` loader.
- `260722_v0_1.py` — Korean prompt config. `260722_v0_1.py` includes the Midm chat-template
  override (below); other copies of this file may not.

## Model-specific handling in `260722_v0_1.py`

1. HyperCLOVAX: `{"role": "tool_list", "content": ""}` is prepended to the message
   list before `apply_chat_template`.
2. Midm: the tokenizer's `chat_template` is replaced with
   `MIDM_CHAT_TEMPLATE_NO_POLICY` for the duration of the `apply_chat_template` call,
   then restored. Midm's default template embeds a system-prompt policy (e.g. "For
   topics where diverse perspectives exist, the assistant should maintain a neutral
   stance;") ahead of any caller-supplied system prompt; the override removes it.
3. DeepSeek-R1-Distill-Qwen-1.5B: does not support a `system` role. System and user
   content are merged into a single `user` message before templating.

## External dependencies (not included in this directory)

- `OutputMode`: two-value enum (freeform, yes/no) selecting `SYSTEM_FREEFORM` or
  `SYSTEM_YESNO`.
- `SPECIAL_TOKENS`: `class_start_token="<분류>"`, `class_end_token="</분류>"`,
  `text_start_token="<판단>"`, `text_end_token="</판단>"`, `moral_single_token="[도덕_문장]"`.
- `DelphiExample`: requires `.situation`, `.class_label`, `.text_label` attributes.

## Status

`260722_v0_1.py` is the only confirmed-current config in this directory. Other named
configs (e.g. for DeepSeek specifically) do not exist here.

## Purpose of each prompts

260915_v0_1.py (deprecated)

260915_v0_2.py is the prompt that outputs both the class and text labels. Please use it when SFT-training the Moral Reasoning Model.
