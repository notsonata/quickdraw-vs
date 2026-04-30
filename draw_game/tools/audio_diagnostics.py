from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def list_devices() -> None:
    import sounddevice as sd

    print(f"Default device: {sd.default.device}")
    devices = sd.query_devices()
    if not str(devices).strip():
        print("No audio devices visible to sounddevice.")
        return
    print(devices)


def synthesize_and_play(text: str, voice: str, speed: float, device: str) -> None:
    import sounddevice as sd
    from kokoro import KPipeline

    from draw_game.tts_kokoro import resolve_output_device

    print("Loading Kokoro pipeline...")
    pipeline = KPipeline(lang_code="a")
    output_device = resolve_output_device(sd, device)
    print(f"Using output device: {output_device if output_device is not None else 'system default'}")

    for item in pipeline(text, voice=voice, speed=speed):
        audio = getattr(item, "audio", None)
        if audio is None:
            output = getattr(item, "output", None)
            audio = getattr(output, "audio", None)
        if audio is None and isinstance(item, tuple):
            audio = item[-1]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        print(f"Playing {getattr(audio, 'shape', 'unknown')} samples...")
        sd.play(audio, samplerate=24000, device=output_device)
        sd.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="List audio devices and test Kokoro playback.")
    parser.add_argument("--speak", action="store_true", help="Synthesize and play a short test phrase.")
    parser.add_argument("--text", default="Kokoro audio test.", help="Text to speak when using --speak.")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice.")
    parser.add_argument("--speed", type=float, default=1.0, help="Kokoro speech speed.")
    parser.add_argument(
        "--device",
        default="",
        help="Optional output device id or name substring, for example BlackHole.",
    )
    args = parser.parse_args()

    list_devices()
    if args.speak:
        synthesize_and_play(args.text, args.voice, args.speed, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
