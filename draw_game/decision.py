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
        self.last_top1_confidence: float | None = None
        self.confidence_stall_count = 0
        self.top1_spoken_for_current_label = False
        self.top1_repeat_count = 0
        self.last_label_since = 0.0
        self.low_confidence_since = 0.0
        self.last_spoken_label: str | None = None
        self.spoken_label_counts: dict[str, int] = {}
        self.last_speech_time = -1_000_000.0
        self.last_taunt_time = -1_000_000.0
        self.ai_guesses_this_round = 0
        self.recent_top_candidates: deque[list[list[str | float]]] = deque(maxlen=5)
        self.recent_spoken_labels: deque[str] = deque(maxlen=3)

    def start_round(self) -> None:
        now = self._now()
        self.round_active = True
        self.round_start_time = now
        self.last_top1_label = None
        self.last_top1_confidence = None
        self.confidence_stall_count = 0
        self.top1_spoken_for_current_label = False
        self.top1_repeat_count = 0
        self.last_label_since = now
        self.low_confidence_since = now
        self.last_spoken_label = None
        self.spoken_label_counts = {}
        self.last_speech_time = -1_000_000.0
        self.last_taunt_time = -1_000_000.0
        self.ai_guesses_this_round = 0
        self.recent_top_candidates.clear()
        self.recent_spoken_labels.clear()

    def end_round(self) -> None:
        self.round_active = False
        self.last_top1_label = None
        self.last_top1_confidence = None
        self.confidence_stall_count = 0
        self.top1_spoken_for_current_label = False
        self.top1_repeat_count = 0
        self.last_spoken_label = None
        self.spoken_label_counts = {}
        self.ai_guesses_this_round = 0
        self.recent_top_candidates.clear()
        self.recent_spoken_labels.clear()

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
        ordered_labels: list[str] = []
        for item in top_candidates[:5]:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            label = str(item[0])
            score = float(item[1])
            if score < settings.AI_MIN_SPOKEN_LABEL_CONFIDENCE or score <= 0.0:
                continue
            if label not in ordered_labels:
                ordered_labels.append(label)

        used_fallback = False
        if (
            repeated_label
            and self.confidence_stall_count > 0
            and getattr(settings, "AI_SPEAK_EVERY_SCAN", False)
            and self.top1_spoken_for_current_label
        ):
            fallback_labels = [label for label in ordered_labels if label != top1 and label not in self.recent_spoken_labels]
            if not fallback_labels:
                fallback_labels = [label for label in ordered_labels if label != top1]
            if fallback_labels:
                fallback_index = (self.confidence_stall_count - 1) % len(fallback_labels)
                primary_label = fallback_labels[fallback_index]
                primary_score = scores.get(primary_label, primary_score)
                alternate_label = top1
                used_fallback = True
        elif repeated_label and confidence < major_confidence:
            fallback_labels = [label for label in ordered_labels if label != self.last_spoken_label and label not in self.recent_spoken_labels]
            if not fallback_labels:
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
                if secondary_label in ordered_labels:
                    alternate_label = secondary_label

        current_second = None
        if len(top_candidates) > 1 and isinstance(top_candidates[1], (list, tuple)) and len(top_candidates[1]) >= 2:
            current_second = str(top_candidates[1][0])
            current_second_confidence = float(top_candidates[1][1])
            if (
                not used_fallback
                and current_second != primary_label
                and current_second_confidence >= max(0.12, confidence * 0.82)
                and current_second_confidence >= settings.AI_MIN_SPOKEN_LABEL_CONFIDENCE
                and current_second_confidence > 0.0
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
        source = str(result.get("source", "quickdraw"))

        if top1 != self.last_top1_label:
            self.last_top1_label = top1
            self.last_top1_confidence = confidence
            self.confidence_stall_count = 0
            self.top1_spoken_for_current_label = False
            self.top1_repeat_count = 0
            self.last_label_since = now
        else:
            self.top1_repeat_count += 1
            if self.last_top1_confidence is not None and abs(confidence - self.last_top1_confidence) <= 1e-4:
                self.confidence_stall_count += 1
            else:
                self.confidence_stall_count = 0
            self.last_top1_confidence = confidence
        spoken_label, alternate_label = self._choose_spoken_labels(top1, confidence, top5)
        stable_for_sec = max(0.0, now - self.last_label_since)
        if confidence <= 0.0 or confidence < settings.AI_MIN_CONFIDENCE:
            if self.low_confidence_since == 0.0:
                self.low_confidence_since = now
        else:
            self.low_confidence_since = now
        low_confidence_for_sec = max(0.0, now - self.low_confidence_since)

        reason = "normal_guess"
        should_speak = True
        speech_kind = "guess"
        repeat_cap = getattr(settings, "AI_MAX_REPEAT_SAME_LABEL_PER_ROUND", 3)
        repeat_cooldown_sec = getattr(settings, "AI_REPEAT_LABEL_COOLDOWN_SEC", 10.0)
        high_confidence_repeat = getattr(settings, "AI_HIGH_CONFIDENCE_REPEAT", 0.85)
        spoken_label_count = self.spoken_label_counts.get(spoken_label, 0)

        if not self.round_active:
            reason = "round_not_active"
            should_speak = False
        elif source == "gemma":
            reason = "gemma_detection"
            should_speak = True
        elif self.ai_guesses_this_round >= settings.MAX_AI_GUESSES_PER_ROUND:
            reason = "max_guesses_per_round"
            should_speak = False
        elif spoken_label_count >= repeat_cap:
            reason = "same_label_repeat_cap"
            should_speak = False
        elif spoken_label == self.last_spoken_label:
            if now - self.last_speech_time < repeat_cooldown_sec:
                reason = "duplicate_label_cooldown"
                should_speak = False
            elif confidence < high_confidence_repeat:
                reason = "low_confidence"
                should_speak = False
            else:
                reason = "same_label_repeat_allowed"
                should_speak = True
        elif now - self.last_speech_time < settings.AI_SPEECH_COOLDOWN_SEC:
            reason = "cooldown"
            should_speak = False
        elif confidence <= 0.0 or confidence < settings.AI_MIN_CONFIDENCE:
            if now - self.last_taunt_time >= getattr(settings, "AI_TAUNT_COOLDOWN_SEC", 5.0):
                reason = "low_confidence_taunt"
                speech_kind = "taunt"
            else:
                reason = "low_confidence"
                should_speak = False
        elif getattr(settings, "AI_SPEAK_EVERY_SCAN", False):
            if self.confidence_stall_count > 0 and spoken_label != top1:
                reason = "stalled_confidence_top5"
            elif self.last_spoken_label == spoken_label and spoken_label != top1:
                reason = "top5_fallback"
            else:
                reason = "speak_every_scan"
        elif now - self.round_start_time < settings.AI_FIRST_GUESS_DELAY_SEC:
            reason = "first_guess_delay"
            should_speak = False
        elif stable_for_sec < settings.AI_STABLE_FOR_SEC:
            reason = "not_stable"
            should_speak = False

        if should_speak:
            if speech_kind == "taunt":
                self.last_taunt_time = now
            else:
                self.last_spoken_label = spoken_label
                self.spoken_label_counts[spoken_label] = spoken_label_count + 1
                self.recent_spoken_labels.append(spoken_label)
                if spoken_label == top1:
                    self.top1_spoken_for_current_label = True
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
            "source": source,
            "interrupt_current": should_speak and speech_kind == "guess",
            "ai_guesses_this_round": self.ai_guesses_this_round,
        }
