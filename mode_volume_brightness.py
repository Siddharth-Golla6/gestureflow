# ─────────────────────────────────────────────
#  mode_volume_brightness.py
# ─────────────────────────────────────────────

import platform
import numpy as np
import cv2
from collections import deque
from hand_tracker import HandTracker, THUMB_TIP, INDEX_TIP
from config import VOL_BRIGHT_MIN_DIST, VOL_BRIGHT_MAX_DIST, SMOOTHING_WINDOW, BRIGHTNESS_MIN_CHANGE

_OS = platform.system()

# ── Volume backend ────────────────────────────────────────────────────────────

def _init_pycaw():
    """Return (volume_ctrl, vol_range) or raise."""
    import comtypes
    comtypes.CoInitialize()
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    speakers = AudioUtilities.GetSpeakers()

    # Newer pycaw (>=0.10) wraps IMMDevice in an AudioDevice Python object.
    # Older pycaw returns the raw COM IMMDevice directly.
    # Use ._dev when the wrapper is present.
    raw_dev = getattr(speakers, '_dev', speakers)

    interface   = raw_dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
    vol_range   = volume_ctrl.GetVolumeRange()   # (min_dB, max_dB, step)
    return volume_ctrl, vol_range

if _OS == "Windows":
    try:
        _volume_ctrl, _vol_range = _init_pycaw()
        PYCAW_OK = True
        print("[Volume] pycaw ready.")
    except Exception as e:
        print(f"[Volume] pycaw failed ({e}). Volume bar visible but control disabled.")
        PYCAW_OK = False

    def set_system_volume(level_0_100: float):
        if not PYCAW_OK:
            return
        try:
            # SetMasterVolumeLevelScalar expects 0.0–1.0 (linear, feels natural)
            scalar = float(np.clip(level_0_100 / 100.0, 0.0, 1.0))
            _volume_ctrl.SetMasterVolumeLevelScalar(scalar, None)
        except Exception as e:
            print(f"[Volume] set error: {e}")

    def get_system_volume() -> float:
        if not PYCAW_OK:
            return 50.0
        try:
            return float(_volume_ctrl.GetMasterVolumeLevelScalar()) * 100.0
        except Exception:
            return 50.0

elif _OS == "Darwin":
    import subprocess

    def set_system_volume(level_0_100: float):
        subprocess.run(["osascript", "-e",
                        f"set volume output volume {int(level_0_100)}"],
                       capture_output=True)

    def get_system_volume() -> float:
        r = subprocess.run(["osascript", "-e",
                            "output volume of (get volume settings)"],
                           capture_output=True, text=True)
        try:
            return float(r.stdout.strip())
        except Exception:
            return 50.0

else:  # Linux
    import subprocess

    def set_system_volume(level_0_100: float):
        subprocess.run(["amixer", "-q", "sset", "Master", f"{int(level_0_100)}%"],
                       capture_output=True)

    def get_system_volume() -> float:
        r = subprocess.run(["amixer", "get", "Master"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "%" in line:
                try:
                    return float(line.split("[")[1].split("%")[0])
                except Exception:
                    pass
        return 50.0

# ── Brightness backend ────────────────────────────────────────────────────────

try:
    import screen_brightness_control as sbc
    SBC_OK = True
except ImportError:
    SBC_OK = False
    print("[Brightness] screen_brightness_control not installed. Brightness disabled.")

def set_brightness(level_0_100: float):
    if SBC_OK:
        try:
            sbc.set_brightness(int(level_0_100))
        except Exception:
            pass

def get_brightness() -> float:
    if SBC_OK:
        try:
            b = sbc.get_brightness()
            return float(b[0] if isinstance(b, list) else b)
        except Exception:
            pass
    return 50.0


# ── Controller ────────────────────────────────────────────────────────────────

class VolumeBrightnessMode:
    """
    Thumb↔index pinch → volume or brightness.
    Two hands: left=volume, right=brightness.
    One hand:  press v/b to choose.
    Distances normalised by hand_size → works at any camera distance.
    """

    def __init__(self):
        self._smooth_vol  = deque(maxlen=SMOOTHING_WINDOW)
        self._smooth_bri  = deque(maxlen=SMOOTHING_WINDOW)
        self.sub_mode     = "volume"
        self._current_vol = get_system_volume()
        self._current_bri = get_brightness()
        self._last_set_bri = self._current_bri   # track last value actually sent

    def toggle_sub_mode(self):
        self.sub_mode = "brightness" if self.sub_mode == "volume" else "volume"

    def _pinch_to_level(self, hand_lms, tracker: HandTracker) -> float:
        """
        Scale-invariant pinch: distance / hand_size, then map to 0-100.
        """
        raw   = tracker.dist_norm(hand_lms, THUMB_TIP, INDEX_TIP)
        scale = max(tracker.hand_size(hand_lms), 1e-4)
        norm  = raw / scale   # typically ~0.1 (closed) to ~1.5 (open)

        # Clamp and map to 0-100
        lo, hi = VOL_BRIGHT_MIN_DIST, VOL_BRIGHT_MAX_DIST
        clamped = max(lo, min(hi, norm))
        return (clamped - lo) / (hi - lo) * 100.0

    def process(self, frame: np.ndarray, results, tracker: HandTracker) -> np.ndarray:
        h, w = frame.shape[:2]
        vol_level = None
        bri_level = None

        if results.multi_hand_landmarks:
            hand_list   = results.multi_hand_landmarks
            handed_list = results.multi_handedness

            for hand_lms, handedness in zip(hand_list, handed_list):
                label = handedness.classification[0].label
                level = self._pinch_to_level(hand_lms, tracker)

                # Draw pinch line
                tx, ty = tracker.landmark_px(hand_lms, THUMB_TIP, w, h)
                ix, iy = tracker.landmark_px(hand_lms, INDEX_TIP,  w, h)
                cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 255), 2)
                cv2.circle(frame, (tx, ty), 7, (0, 255, 200), cv2.FILLED)
                cv2.circle(frame, (ix, iy), 7, (0, 255, 200), cv2.FILLED)

                if len(hand_list) == 2:
                    if label == "Left":
                        vol_level = level
                    else:
                        bri_level = level
                else:
                    if self.sub_mode == "volume":
                        vol_level = level
                    else:
                        bri_level = level

        if vol_level is not None:
            self._smooth_vol.append(vol_level)
            self._current_vol = float(np.mean(self._smooth_vol))
            set_system_volume(self._current_vol)

        if bri_level is not None:
            self._smooth_bri.append(bri_level)
            smoothed = float(np.mean(self._smooth_bri))
            self._current_bri = smoothed
            # Only call set_brightness when change is large enough to matter
            # (avoids hammering the display driver every frame for tiny tremors)
            if abs(smoothed - self._last_set_bri) >= BRIGHTNESS_MIN_CHANGE:
                set_brightness(smoothed)
                self._last_set_bri = smoothed

        self._draw_ui(frame)
        return frame

    def _draw_ui(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        bar_h = int(h * 0.55)
        bx    = w - 70

        self._draw_bar(frame, bx - 35, h // 5, bar_h,
                       self._current_vol, (0, 200, 255), "VOL")
        self._draw_bar(frame, bx + 5,  h // 5, bar_h,
                       self._current_bri, (255, 220, 50), "BRI")

        active = "VOLUME" if self.sub_mode == "volume" else "BRIGHTNESS"
        cv2.rectangle(frame, (0, h - 50), (420, h), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"Single hand controls: {active}",
                    (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
        cv2.putText(frame, "[v]=Volume   [b]=Brightness   Two hands=both",
                    (10, h - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    @staticmethod
    def _draw_bar(frame, x, y_top, height, level, colour, label):
        filled = int(height * level / 100)
        cv2.rectangle(frame, (x, y_top), (x + 28, y_top + height),
                      (40, 40, 40), cv2.FILLED)
        if filled > 0:
            cv2.rectangle(frame, (x, y_top + height - filled),
                          (x + 28, y_top + height), colour, cv2.FILLED)
        cv2.rectangle(frame, (x, y_top), (x + 28, y_top + height),
                      (180, 180, 180), 1)
        cv2.putText(frame, f"{int(level)}%", (x - 5, y_top + height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        cv2.putText(frame, label, (x - 2, y_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
