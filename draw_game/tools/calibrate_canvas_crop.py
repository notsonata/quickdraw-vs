from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency
    cv2 = None

try:
    import mss
except Exception:  # pragma: no cover - runtime dependency
    mss = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / "draw_game" / ".env"


def compute_bounding_box(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    if len(points) != 4:
        raise ValueError("Exactly four points are required.")
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    if w <= 0 or h <= 0:
        raise ValueError("Selected points must form a non-empty rectangle.")
    return x, y, w, h


def update_env_crop(env_path: Path, x: int, y: int, w: int, h: int) -> None:
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = existing.splitlines()
    replacements = {
        "CANVAS_X": f"CANVAS_X={x}",
        "CANVAS_Y": f"CANVAS_Y={y}",
        "CANVAS_W": f"CANVAS_W={w}",
        "CANVAS_H": f"CANVAS_H={h}",
    }

    seen: set[str] = set()
    updated_lines: list[str] = []
    for line in lines:
        replaced = False
        for key, value in replacements.items():
            if line.startswith(f"{key}="):
                updated_lines.append(value)
                seen.add(key)
                replaced = True
                break
        if not replaced:
            updated_lines.append(line)

    for key in ("CANVAS_X", "CANVAS_Y", "CANVAS_W", "CANVAS_H"):
        if key not in seen:
            updated_lines.append(replacements[key])

    output = "\n".join(updated_lines).rstrip() + "\n"
    env_path.write_text(output, encoding="utf-8")


def capture_fullscreen() -> np.ndarray:
    if mss is None:
        raise RuntimeError("mss is not installed. Run pip install -r requirements.txt.")
    with mss.mss() as screenshotter:
        monitor = screenshotter.monitors[1]
        raw = screenshotter.grab(monitor)
    frame = np.asarray(raw)
    return frame[:, :, :3][:, :, ::-1].copy()


def draw_overlay(frame: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    canvas = frame.copy()
    for index, point in enumerate(points, start=1):
        cv2.circle(canvas, point, 8, (0, 0, 255), thickness=-1)
        cv2.putText(
            canvas,
            str(index),
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if len(points) == 4:
        x, y, w, h = compute_bounding_box(points)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return canvas


def run_calibration(env_path: Path = ENV_PATH) -> int:
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed. Run pip install -r requirements.txt.")

    frame = capture_fullscreen()
    points: list[tuple[int, int]] = []
    window_name = "Calibrate Canva Crop"

    instructions = [
        "Click the four Canva corners in any order.",
        "Press r to reset, Enter to save, q or Esc to quit.",
    ]
    print("\n".join(instructions))

    def handle_click(event, x, y, flags, param):  # pragma: no cover - UI callback
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x), int(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, handle_click)

    try:
        while True:
            preview = draw_overlay(frame, points)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF

            if key in {27, ord("q")}:
                print("Calibration cancelled.")
                return 1
            if key == ord("r"):
                points.clear()
            if key in {13, 10} and len(points) == 4:
                break
    finally:
        cv2.destroyAllWindows()

    x, y, w, h = compute_bounding_box(points)
    update_env_crop(env_path, x, y, w, h)
    print(f"Saved crop to {env_path}")
    print(f"CANVAS_X={x}")
    print(f"CANVAS_Y={y}")
    print(f"CANVAS_W={w}")
    print(f"CANVAS_H={h}")
    return 0


def main() -> int:
    env_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ENV_PATH
    return run_calibration(env_path)


if __name__ == "__main__":
    raise SystemExit(main())
