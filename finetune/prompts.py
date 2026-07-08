"""Instruction-style prompt formatting for the classification task.

Even though the underlying model uses a classification head, we still format the
raw text into a consistent instruction prompt. This keeps inputs uniform across
runs and matches how the same data would be fed to a generative model, so the
"prompt formatting" stage is a real, testable part of the pipeline.
"""
from __future__ import annotations

# emotion label ids -> human-readable names (fixed by the dair-ai/emotion dataset)
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

INSTRUCTION = (
    "Classify the primary emotion expressed in the message. "
    "Choose exactly one of: sadness, joy, love, anger, fear, surprise."
)

PROMPT_TEMPLATE = "{instruction}\n\nMessage: {text}\n\nEmotion:"


def format_prompt(text: str, use_prompt_formatting: bool = True) -> str:
    """Wrap raw text in the instruction template (or return it unchanged)."""
    text = (text or "").strip()
    if not use_prompt_formatting:
        return text
    return PROMPT_TEMPLATE.format(instruction=INSTRUCTION, text=text)


def id_to_name(label_id: int) -> str:
    return LABEL_NAMES[int(label_id)]


def num_labels() -> int:
    return len(LABEL_NAMES)
