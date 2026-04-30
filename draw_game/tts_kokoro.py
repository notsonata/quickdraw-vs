from __future__ import annotations

import threading
from typing import Optional

import numpy as np

try:
    from .config import settings
except ImportError:  # pragma: no cover
    from config import settings


class TTSWorker:
    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._pending_text: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self._pyttsx3 = None
        self._kokoro_pipeline = None
        self._output_stream = None
        self._output_stream_key: tuple[object | None, int, int] | None = None

    def speak(self, text: str) -> None:
        if not settings.TTS_ENABLED:
            print(f"[TTS disabled] {text}")
            return
        self._ensure_started()
        with self._lock:
            self._pending_text = text
            self._lock.notify()

    def prime(self) -> None:
        if not settings.TTS_ENABLED:
            return
        try:
            self._warmup_now()
        except Exception:
            pass

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._thread.start()
            self._started = True

    def _run(self) -> None:
        while True:
            with self._lock:
                while self._pending_text is None:
                    self._lock.wait()
                text = self._pending_text
                self._pending_text = None
            try:
                self._speak_now(text)
            except Exception as exc:
                print(f"[TTS fallback] {text} (reason: {exc})")

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

    def _warmup_now(self) -> None:
        pipeline = self._get_kokoro_pipeline()
        if pipeline is None:
            return
        try:
            import sounddevice as sd
        except Exception:
            return
        device = resolve_output_device(sd, settings.KOKORO_AUDIO_DEVICE)
        samplerate = get_output_samplerate(sd, device)
        prepare_stream_audio(
            sd,
            np.zeros(max(1, samplerate // 20), dtype=np.float32),
            device=device,
            samplerate=samplerate,
            padding_ms=0,
        )

    def _play_audio(self, audio: object) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError("sounddevice is unavailable for Kokoro playback") from exc
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        device = resolve_output_device(sd, settings.KOKORO_AUDIO_DEVICE)
        playback_audio, samplerate = prepare_playback_audio(sd, audio, device=device, source_rate=24000)
        stream_audio = prepare_stream_audio(sd, playback_audio, device=device, samplerate=samplerate)
        stream = self._ensure_output_stream(sd, device=device, samplerate=samplerate, channels=stream_audio.shape[1])
        stream.write(stream_audio)

    def _ensure_output_stream(self, sounddevice_module, device: object | None, samplerate: int, channels: int):
        key = (device, samplerate, channels)
        if self._output_stream is not None and self._output_stream_key == key:
            return self._output_stream

        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                pass

        stream = sounddevice_module.OutputStream(
            samplerate=samplerate,
            device=device,
            channels=channels,
            dtype="float32",
            blocksize=0,
            latency="high",
        )
        stream.start()
        self._output_stream = stream
        self._output_stream_key = key
        return stream

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


def prime() -> None:
    _WORKER.prime()


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


def get_output_samplerate(sounddevice_module, device: object | None) -> int:
    if device is None:
        default_output = sounddevice_module.default.device[1]
        if default_output is None or int(default_output) < 0:
            return 24000
        device = int(default_output)

    info = sounddevice_module.query_devices(device)
    samplerate = int(round(float(info.get("default_samplerate", 24000))))
    return samplerate if samplerate > 0 else 24000


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate <= 0 or target_rate <= 0 or samples.size == 0:
        return samples
    if source_rate == target_rate:
        return samples

    duration = samples.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
    target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=True)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def prepare_playback_audio(
    sounddevice_module,
    audio: object,
    device: object | None,
    source_rate: int,
) -> tuple[np.ndarray, int]:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    target_rate = get_output_samplerate(sounddevice_module, device)
    if target_rate != source_rate:
        samples = resample_audio(samples, source_rate, target_rate)
    return samples, target_rate


def prepare_stream_audio(
    sounddevice_module,
    audio: object,
    device: object | None,
    samplerate: int,
    padding_ms: int = 250,
) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    padding_samples = max(1, int(round(samplerate * (padding_ms / 1000.0))))
    padded = np.concatenate(
        [
            np.zeros(padding_samples, dtype=np.float32),
            samples,
            np.zeros(padding_samples, dtype=np.float32),
        ]
    )

    info = sounddevice_module.query_devices(device)
    channels = int(info.get("max_output_channels", 1))
    channels = 2 if channels >= 2 else 1
    if channels == 1:
        return padded[:, None]
    stereo = np.repeat(padded[:, None], channels, axis=1)
    return stereo
