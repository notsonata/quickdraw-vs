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

HEDGED_LINES = [
    "{label} or {alternate}.",
    "Maybe {label} or {alternate}.",
    "{label}... or {alternate}.",
    "Is it {label} or {alternate}?",
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


def make_spoken_line(label: str, confidence: float, alternate_label: str | None = None) -> str:
    clean_label = label.replace("_", " ").strip()
    clean_alternate = (alternate_label or "").replace("_", " ").strip()
    if clean_alternate and clean_alternate != clean_label:
        return random.choice(HEDGED_LINES).format(label=clean_label, alternate=clean_alternate)
    templates = HIGH_CONFIDENCE_LINES if confidence >= 0.85 else MEDIUM_CONFIDENCE_LINES
    return random.choice(templates).format(label=clean_label)


def make_low_confidence_taunt() -> str:
    return random.choice(LOW_CONFIDENCE_TAUNTS)
