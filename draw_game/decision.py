from __future__ import annotations

import time
from typing import Callable

try:
    from .config import settings
except ImportError:  # pragma: no cover
    from config import settings


class SpeechGate:
    def __init__(self, now_func: Callable[[], float] | None = None) -> None:
        self._now = now_func or time.monotonic
        self.round_active = False
        self.round_start_time = 0.0
        self.last_top1_label: str | None = None
        self.last_label_since = 0.0
        self.last_spoken_label: str | None = None
        self.last_speech_time = -1_000_000.0
        self.ai_guesses_this_round = 0

    def start_round(self) -> None:
        now = self._now()
        self.round_active = True
        self.round_start_time = now
        self.last_top1_label = None
        self.last_label_since = now
        self.last_spoken_label = None
        self.last_speech_time = -1_000_000.0
        self.ai_guesses_this_round = 0

    def end_round(self) -> None:
        self.round_active = False

    def update(self, result: dict) -> dict:
        now = self._now()
        top1 = str(result.get("top1", ""))
        confidence = float(result.get("confidence", 0.0))
        top3 = result.get("top3", [])

        if top1 != self.last_top1_label:
            self.last_top1_label = top1
            self.last_label_since = now
        stable_for_sec = max(0.0, now - self.last_label_since)

        reason = "stable_confident_guess"
        should_speak = True

        if not self.round_active:
            reason = "round_not_active"
            should_speak = False
        elif now - self.round_start_time < settings.AI_FIRST_GUESS_DELAY_SEC:
            reason = "first_guess_delay"
            should_speak = False
        elif confidence < settings.AI_MIN_CONFIDENCE:
            reason = "low_confidence"
            should_speak = False
        elif stable_for_sec < settings.AI_STABLE_FOR_SEC:
            reason = "not_stable"
            should_speak = False
        elif now - self.last_speech_time < settings.AI_SPEECH_COOLDOWN_SEC:
            reason = "cooldown"
            should_speak = False
        elif self.ai_guesses_this_round >= settings.MAX_AI_GUESSES_PER_ROUND:
            reason = "max_guesses_reached"
            should_speak = False
        elif top1 == self.last_spoken_label:
            reason = "duplicate_label"
            should_speak = False

        if should_speak:
            self.last_spoken_label = top1
            self.last_speech_time = now
            self.ai_guesses_this_round += 1

        return {
            "round_active": self.round_active,
            "top1": top1,
            "confidence": round(confidence, 4),
            "top3": top3,
            "stable_ms": int(round(stable_for_sec * 1000)),
            "should_speak": should_speak,
            "reason": reason,
            "ai_guesses_this_round": self.ai_guesses_this_round,
        }
