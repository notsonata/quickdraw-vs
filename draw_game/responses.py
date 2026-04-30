from __future__ import annotations

import random


HIGH_CONFIDENCE_LINES = [
    "My guess is {label}.",
    "I think it is {label}.",
    "I am locking in {label}.",
    "That looks like {label}.",
    "{label}. Final answer.",
]

MEDIUM_CONFIDENCE_LINES = [
    "Maybe {label}.",
    "I am guessing {label}.",
    "Could be {label}.",
    "My current guess is {label}.",
]


def make_spoken_line(label: str, confidence: float) -> str:
    clean_label = label.replace("_", " ").strip()
    templates = HIGH_CONFIDENCE_LINES if confidence >= 0.85 else MEDIUM_CONFIDENCE_LINES
    return random.choice(templates).format(label=clean_label)
