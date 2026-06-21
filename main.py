# ─────────────────────────────────────────────
#  main.py  —  Shared camera loop + mode switching
# ─────────────────────────────────────────────

import cv2
import time
import numpy as np

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    KEY_MODE_VOLUME, KEY_MODE_MUSIC, KEY_MODE_MOUSE,
    KEY_PAUSE, KEY_VOLUME_SUB, KEY_BRIGHTNESS,
    THUMBSUP_HOLD_SEC,
)
from hand_tracker import HandTracker
from mode_volume_brightness import VolumeBrightnessMode
from mode_music_player      import MusicPlayerMode
from mode_virtual_mouse     import VirtualMouseMode

_THSEC = THUMBSUP_HOLD_SEC


def _open_camera():
    """
    Find a working camera. Tries every index 0-9 with both DirectShow and
    the default backend. Camo virtual camera needs several warm-up reads
    before it starts delivering real frames.
    """
    # List device names if pygrabber is available
    camo_index = None
    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
        print("[Camera] Devices found:")
        for i, name in enumerate(devices):
            print(f"  [{i}] {name}")
            if "camo" in name.lower():
                camo_index = i
    except Exception:
        pass

    # Scan order: Camo first if identified, then 0-9
    order = []
    if camo_index is not None:
        order.append((camo_index, cv2.CAP_DSHOW))
    for idx in range(10):
        if idx != camo_index:
            order.append((idx, cv2.CAP_DSHOW))
    # Also try without DSHOW (Media Foundation) for Camo USB mode
    if camo_index is not None:
        order.append((camo_index, cv2.CAP_MSMF))

    for idx, backend in order:
        try:
            c = cv2.VideoCapture(idx, backend)
            if not c.isOpened():
                c.release()
                continue

            # Set resolution BEFORE reading — Camo needs this first
            c.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            c.set(cv2.CAP_PROP_FPS, 30)
            # Disable internal buffer so we always get the latest frame
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warm-up: Camo virtual driver needs several reads to start streaming
            good = False
            for _ in range(30):
                ok, frame = c.read()
                if ok and frame is not None and frame.shape[0] > 0 and frame.sum() > 1000:
                    good = True
                    break

            if good:
                backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else "MSMF"
                label = f"Camo [{backend_name}]" if idx == camo_index else f"index {idx} [{backend_name}]"
                print(f"[Camera] Using {label}")
                return c

            c.release()
        except Exception as e:
            print(f"[Camera] index {idx} error: {e}")

    return None

# ── Mode registry ─────────────────────────────
MODES = {
    1: "Volume/Brightness",
    2: "Music Player",
    3: "Virtual Mouse",
}
MODE_COLOURS = {
    1: (0, 200, 255),
    2: (0, 255, 120),
    3: (255, 150, 0),
}


def draw_mode_hud(frame: np.ndarray, mode: int, paused: bool, fps: float):
    h, w = frame.shape[:2]
    colour = MODE_COLOURS[mode]

    # Mode badge
    label = f"MODE {mode}: {MODES[mode]}"
    cv2.rectangle(frame, (0, 0), (len(label) * 14 + 20, 38), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)

    # Paused overlay
    if paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, "⏸  PAUSED  (SPACE to resume)",
                    (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 255), 3)

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Key hints at bottom
    hints = "[1]Vol/Bri  [2]Music  [3]Mouse  [SPACE]Pause  [Q]Quit"
    cv2.putText(frame, hints, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)


def main():
    tracker = HandTracker()
    modes   = {
        1: VolumeBrightnessMode(),
        2: MusicPlayerMode(),
        3: VirtualMouseMode(),
    }
    current_mode = 1
    paused       = False

    # Thumbs-up hold timer for gesture-based mode switching
    thumbsup_start = None

    # Enumerate all camera devices and prefer Camo if found
    cap = _open_camera()
    if cap is None:
        print("[ERROR] No camera found. Make sure Camo is running and your phone is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev_ts = time.time()

    print("Hand Gesture Control Suite started.")
    print("Press 1/2/3 to switch modes, SPACE to pause, Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel

        # FPS
        now    = time.time()
        fps    = 1.0 / max(now - prev_ts, 1e-6)
        prev_ts = now

        # Run hand tracking (always)
        results = tracker.process(frame)
        tracker.draw_landmarks(frame, results)

        # ── Gesture-based mode cycle (thumbs-up held) ─────────────────
        if results.multi_hand_landmarks and not paused:
            hand_lms   = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            fingers    = tracker.fingers_up(hand_lms, handedness)
            # Require thumb tip above wrist so a sideways thumb doesn't trigger
            thumb_above_wrist = (hand_lms.landmark[4].y < hand_lms.landmark[0].y)
            is_thumbsup = fingers[0] and not any(fingers[1:]) and thumb_above_wrist

            if is_thumbsup:
                if thumbsup_start is None:
                    thumbsup_start = now
                elif (now - thumbsup_start) >= _THSEC:
                    current_mode   = (current_mode % 3) + 1
                    thumbsup_start = None
                else:
                    # Draw hold progress ring
                    progress = (now - thumbsup_start) / _THSEC
                    h, w = frame.shape[:2]
                    cx, cy = w // 2, 60
                    cv2.ellipse(frame, (cx, cy), (30, 30), -90,
                                0, int(360 * progress), (0, 255, 200), 3)
                    cv2.putText(frame, "Hold for mode switch",
                                (cx - 100, cy + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
            else:
                thumbsup_start = None

        # ── Dispatch to active mode handler ───────────────────────────
        if not paused:
            frame = modes[current_mode].process(frame, results, tracker)

        # ── HUD overlay ───────────────────────────────────────────────
        draw_mode_hud(frame, current_mode, paused, fps)

        cv2.imshow("Hand Gesture Control", frame)

        # ── Key handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == KEY_PAUSE:
            paused = not paused
        elif key == KEY_MODE_VOLUME:
            current_mode = 1
        elif key == KEY_MODE_MUSIC:
            current_mode = 2
        elif key == KEY_MODE_MOUSE:
            current_mode = 3
        elif key == KEY_VOLUME_SUB or key == KEY_BRIGHTNESS:
            if current_mode == 1:
                modes[1].toggle_sub_mode()

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
