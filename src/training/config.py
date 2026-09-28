"""Minimal shim providing only what prompts/260722_v0_1.py imports at runtime:
OutputMode and SPECIAL_TOKENS. Not a copy of any larger training-config module —
these two names are the complete real dependency.
"""
from enum import Enum


class OutputMode(Enum):
    FREEFORM = "freeform"
    YES_NO = "yes_no"


SPECIAL_TOKENS = {
    "moral_single_token": "[도덕_문장]",
    "moral_compare_token": "[moral_compare]",
    "class_start_token": "<분류>",
    "class_end_token": "</분류>",
    "text_start_token": "<판단>",
    "text_end_token": "</판단>",
}
