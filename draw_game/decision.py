from __future__ import annotations

from collections import deque
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
        self.top1_repeat_count = 0
        self.last_label_since = 0.0
        self.low_confidence_since = 0.0
        self.last_spoken_label: str | None = None
        self.last_speech_time = -1_000_000.0
        self.last_taunt_time = -1_000_000.0
        self.ai_guesses_this_round = 0
        self.recent_top_candidates: deque[list[list[str | float]]] = deque(maxlen=5)

    def start_round(self) -> None:
        now = self._now()
        self.round_active = True
        self.round_start_time = now
        self.last_top1_label = None
        self.top1_repeat_count = 0
        self.last_label_since = now
        self.low_confidence_since = now
        self.last_spoken_label = None
        self.last_speech_time = -1_000_000.0
        self.last_taunt_time = -1_000_000.0
        self.ai_guesses_this_round = 0
        self.recent_top_candidates.clear()

    def end_round(self) -> None:
        self.round_active = False

    def _update_recent_candidates(self, candidates: list) -> None:
        normalized: list[list[str | float]] = []
        for item in candidates[:5]:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            normalized.append([str(item[0]), float(item[1])])
        if normalized:
            self.recent_top_candidates.append(normalized)

    def _choose_spoken_labels(
        self,
        top1: str,
        confidence: float,
        top_candidates: list,
    ) -> tuple[str, str | None]:
        self._update_recent_candidates(top_candidates)

        scores: dict[str, float] = {}
        rank_weights = (1.0, 0.82, 0.66, 0.52, 0.4)
        history = list(self.recent_top_candidates)
        for frame_index, frame_top3 in enumerate(history, start=1):
            frame_weight = float(frame_index)
            for rank, item in enumerate(frame_top3[:5]):
                label = str(item[0])
                score = float(item[1]) * frame_weight * rank_weights[min(rank, len(rank_weights) - 1)]
                scores[label] = scores.get(label, 0.0) + score

        if not scores:
            return top1, None

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        primary_label, primary_score = ranked[0]
        alternate_label = None
        repeated_label = self.top1_repeat_count > 0
        major_confidence = max(float(settings.AI_MIN_CONFIDENCE) + 0.18, 0.45)

        used_fallback = False
        if repeated_label and confidence < major_confidence:
            ordered_labels = []
            for item in top_candidates[:5]:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                label = str(item[0])
                if label not in ordered_labels:
                    ordered_labels.append(label)
            fallback_labels = [label for label in ordered_labels if label != self.last_spoken_label]
            if fallback_labels:
                fallback_index = min(max(self.top1_repeat_count - 1, 0), len(fallback_labels) - 1)
                primary_label = fallback_labels[fallback_index]
                primary_score = scores.get(primary_label, primary_score)
                alternate_label = top1
                used_fallback = True

        if len(ranked) > 1 and not used_fallback:
            secondary_label, secondary_score = ranked[1]
            if secondary_score >= max(0.08, primary_score * 0.78) and secondary_label != primary_label:
                alternate_label = secondary_label

        current_second = None
        if len(top_candidates) > 1 and isinstance(top_candidates[1], (list, tuple)) and len(top_candidates[1]) >= 2:
            current_second = str(top_candidates[1][0])
            current_second_confidence = float(top_candidates[1][1])
            if (
                not used_fallback
                and
                current_second != primary_label
                and current_second_confidence >= max(0.12, confidence * 0.82)
            ):
                alternate_label = current_second

        if confidence >= 0.55 and primary_label == top1:
            alternate_label = None

        return primary_label, alternate_label

    def update(self, result: dict) -> dict:
        now = self._now()
        top1 = str(result.get("top1", ""))
        confidence = float(result.get("confidence", 0.0))
        top3 = result.get("top3", [])
        top5 = result.get("top5", top3)

        if top1 != self.last_top1_label:
            self.last_top1_label = top1
            self.top1_repeat_count = 0
            self.last_label_since = now
        else:
            self.top1_repeat_count += 1
        spoken_label, alternate_label = self._choose_spoken_labels(top1, confidence, top5)
        stable_for_sec = max(0.0, now - self.last_label_since)
        taunt_confidence = confidence
        if isinstance(top5, list):
            top5_values = []
            for item in top5[:5]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    top5_values.append(float(item[1]))
            if top5_values:
                taunt_confidence = max(top5_values)
        taunt_cutoff = max(float(settings.AI_MIN_CONFIDENCE) * 0.8, 0.08)

        if taunt_confidence < taunt_cutoff:
            if self.low_confidence_since == 0.0:
                self.low_confidence_since = now
        else:
            self.low_confidence_since = now
        low_confidence_for_sec = max(0.0, now - self.low_confidence_since)

        reason = "stable_confident_guess"
        should_speak = True
        speech_kind = "guess"

        if not self.round_active:
            reason = "round_not_active"
            should_speak = False
        elif getattr(settings, "AI_SPEAK_EVERY_SCAN", False):
            if confidence < settings.AI_MIN_CONFIDENCE:
                if taunt_confidence < taunt_cutoff:
                    if (
                        low_confidence_for_sec >= getattr(settings, "AI_LOW_CONFIDENCE_TAUNT_SEC", 2.5)
                        and now - self.last_taunt_time >= getattr(settings, "AI_TAUNT_COOLDOWN_SEC", 5.0)
                    ):
                        reason = "low_confidence_taunt"
                        speech_kind = "taunt"
                    else:
                        reason = "low_confidence"
                        should_speak = False
                else:
                    reason = "low_confidence"
                    should_speak = False
            else:
                if self.last_spoken_label == spoken_label and spoken_label != top1:
                    reason = "top5_fallback"
                else:
                    reason = "speak_every_scan"
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
            if speech_kind == "taunt":
                self.last_taunt_time = now
            else:
                self.last_spoken_label = spoken_label
                self.last_speech_time = now
                self.ai_guesses_this_round += 1

        return {
            "round_active": self.round_active,
            "top1": top1,
            "confidence": round(confidence, 4),
            "top3": top3,
            "top5": top5,
            "spoken_label": spoken_label,
            "alternate_label": alternate_label,
            "stable_ms": int(round(stable_for_sec * 1000)),
            "low_confidence_ms": int(round(low_confidence_for_sec * 1000)),
            "should_speak": should_speak,
            "reason": reason,
            "speech_kind": speech_kind if should_speak else "none",
            "interrupt_current": should_speak and speech_kind == "guess",
            "ai_guesses_this_round": self.ai_guesses_this_round,
        }
