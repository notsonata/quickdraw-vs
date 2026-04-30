from __future__ import annotations

import queue
import threading
from typing import Optional

try:
    from .config import settings
except ImportError:  # pragma: no cover
    from config import settings


class TTSWorker:
    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self._pyttsx3 = None
        self._kokoro_pipeline = None

    def speak(self, text: str) -> None:
        if not settings.TTS_ENABLED:
            print(f"[TTS disabled] {text}")
            return
        if not self._started:
            self._thread.start()
            self._started = True
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            print(f"[TTS busy, skipped] {text}")

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                self._speak_now(text)
            except Exception as exc:
                print(f"[TTS fallback] {text} (reason: {exc})")
            finally:
                self._queue.task_done()

    def _speak_now(self, text: str) -> None:
        if self._try_kokoro(text):
            return
        if self._try_pyttsx3(text):
            return
        print(f"[AI guess] {text}")

    def _try_kokoro(self, text: str) -> bool:
        try:
            pipeline = self._get_kokoro_pipeline()
            if pipeline is None:
                return False
            # Kokoro package APIs have changed across releases. This supports
            # the common generator interface and degrades if playback is absent.
            audio_chunks = pipeline(text, voice=settings.KOKORO_VOICE, speed=settings.KOKORO_SPEED)
            for item in audio_chunks:
                audio = self._extract_audio(item)
                self._play_audio(audio)
            return True
        except Exception as exc:
            print(f"[Kokoro unavailable] {exc}")
            return False

    def _extract_audio(self, item: object) -> object:
        if isinstance(item, tuple):
            return item[-1]
        audio = getattr(item, "audio", None)
        if audio is not None:
            return audio
        output = getattr(item, "output", None)
        audio = getattr(output, "audio", None)
        if audio is not None:
            return audio
        return item

    def _get_kokoro_pipeline(self) -> Optional[object]:
        if self._kokoro_pipeline is not None:
            return self._kokoro_pipeline
        try:
            from kokoro import KPipeline
        except Exception:
            return None
        self._kokoro_pipeline = KPipeline(lang_code="a")
        return self._kokoro_pipeline

    def _play_audio(self, audio: object) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError("sounddevice is unavailable for Kokoro playback") from exc
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        device = resolve_output_device(sd, settings.KOKORO_AUDIO_DEVICE)
        sd.play(audio, samplerate=24000, device=device)
        sd.wait()

    def _try_pyttsx3(self, text: str) -> bool:
        try:
            if self._pyttsx3 is None:
                import pyttsx3

                self._pyttsx3 = pyttsx3.init()
            self._pyttsx3.say(text)
            self._pyttsx3.runAndWait()
            return True
        except Exception:
            return False


_WORKER = TTSWorker()


def speak(text: str) -> None:
    _WORKER.speak(text)


def resolve_output_device(sounddevice_module, configured_device: str = ""):
    configured_device = configured_device.strip()
    if not configured_device:
        return None
    if configured_device.isdigit():
        return int(configured_device)

    devices = sounddevice_module.query_devices()
    matches: list[tuple[int, str]] = []
    for index, device in enumerate(devices):
        name = str(device.get("name", ""))
        output_channels = int(device.get("max_output_channels", 0))
        if output_channels > 0 and configured_device.lower() in name.lower():
            matches.append((index, name))

    if not matches:
        raise RuntimeError(
            f"No output audio device matching KOKORO_AUDIO_DEVICE={configured_device!r}."
        )
    if len(matches) > 1:
        print(f"[Kokoro audio] Multiple device matches for {configured_device!r}: {matches}")
    return matches[0][0]
