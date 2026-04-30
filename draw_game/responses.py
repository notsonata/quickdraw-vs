from __future__ import annotations

import random


HIGH_CONFIDENCE_LINES = [
    "{label}.",
    "It's {label}.",
]

MEDIUM_CONFIDENCE_LINES = [
    "{label}.",
    "I think it's {label}.",
    "Is it {label}?"
]

LOW_CONFIDENCE_TAUNTS = [
    "Draw better.",
    "This is rough.",
    "Holy shit.",
    "Be better please.",
    "You suck at this.",
    "Is this a joke?",
    "My eyes are hurting.",
    "Are you even trying?",
    "I've seen better art from a printer jam.",
    "This is offensive to my training data.",
    "What even is that?",
    "Try harder.",
]


def make_spoken_line(label: str, confidence: float) -> str:
    clean_label = label.replace("_", " ").strip()
    templates = HIGH_CONFIDENCE_LINES if confidence >= 0.85 else MEDIUM_CONFIDENCE_LINES
    return random.choice(templates).format(label=clean_label)


def make_low_confidence_taunt() -> str:
    return random.choice(LOW_CONFIDENCE_TAUNTS)
