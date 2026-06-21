# ─────────────────────────────────────────────
#  mode_music_player.py
# ─────────────────────────────────────────────

import time
import cv2
import numpy as np
import pyautogui
from collections import deque
from hand_tracker import (
    HandTracker,
    WRIST, THUMB_TIP, THUMB_MCP, THUMB_CMC,
    INDEX_MCP,  INDEX_PIP,  INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    RING_MCP,   RING_PIP,   RING_TIP,
    PINKY_MCP,  PINKY_PIP,  PINKY_TIP,
)
from config import MUSIC_ACTION_COOLDOWN_SEC, SWIPE_VELOCITY_THRESHOLD, SWIPE_CONFIRM_FRAMES

pyautogui.FAILSAFE = False

PLAY_PAUSE_KEY = "playpause"
NEXT_KEY       = "nexttrack"
PREV_KEY       = "prevtrack"
VOL_UP_KEY     = "volumeup"
VOL_DOWN_KEY   = "volumedown"


def _send_media_key(key: str):
    try:
        pyautogui.press(key)
    except Exception as e:
        print(f"[MusicPlayer] key error: {e}")


# ── Gesture classifier ────────────────────────────────────────────────────────

class GestureClassifier:
    """
    Angle-based, scale-invariant, rotation-invariant gesture classifier.
    Uses majority voting over a sliding window to eliminate false frames.
    """

    WINDOW = 6      # voting window size
    MIN_VOTES = 5   # must appear at least this many times in the window

    # Thresholds (all scale-invariant — divided by hand_size)
    SPREAD_THRESH    = 0.55   # index↔pinky spread / hand_size for open palm
    THUMBUP_Y_MARGIN = 0.15   # thumb tip must be this far above wrist (/ hand_size)
    THUMBDN_Y_MARGIN = 0.25   # thumb tip below wrist (/ hand_size)

    def __init__(self):
        self._window = deque(maxlen=self.WINDOW)

    def update(self, hand_lms, handedness: str, tracker: HandTracker) -> str:
        raw = self._raw(hand_lms, handedness, tracker)
        self._window.append(raw)
        counts = {}
        for g in self._window:
            counts[g] = counts.get(g, 0) + 1
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.MIN_VOTES else "NONE"

    def _raw(self, hand_lms, handedness: str, tracker: HandTracker) -> str:
        lms   = hand_lms.landmark
        scale = max(tracker.hand_size(hand_lms), 1e-4)
        up    = tracker.fingers_up(hand_lms, handedness)
        # up = [thumb, index, middle, ring, pinky]
        thumb_up  = up[0]
        n_four    = sum(up[1:])   # count of index…pinky extended

        # Check each of the 4 fingers curled via angle
        idx_curled    = tracker.finger_curled(hand_lms, INDEX_MCP,  INDEX_PIP,  INDEX_TIP)
        mid_curled    = tracker.finger_curled(hand_lms, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
        ring_curled   = tracker.finger_curled(hand_lms, RING_MCP,   RING_PIP,   RING_TIP)
        pinky_curled  = tracker.finger_curled(hand_lms, PINKY_MCP,  PINKY_PIP,  PINKY_TIP)
        all_4_curled  = idx_curled and mid_curled and ring_curled and pinky_curled
        three_curled  = sum([idx_curled, mid_curled, ring_curled, pinky_curled]) >= 3

        # ── THUMBS UP ────────────────────────────────────────────────────────
        # Thumb extended upward + all 4 fingers clearly curled
        if thumb_up and all_4_curled:
            dy = (lms[WRIST].y - lms[THUMB_TIP].y) / scale
            if dy > self.THUMBUP_Y_MARGIN:
                return "THUMBS_UP"

        # ── THUMBS DOWN ──────────────────────────────────────────────────────
        # All 4 curled + thumb tip meaningfully BELOW wrist
        if all_4_curled and not thumb_up:
            dy = (lms[THUMB_TIP].y - lms[WRIST].y) / scale
            if dy > self.THUMBDN_Y_MARGIN:
                return "THUMBS_DOWN"

        # ── FIST ─────────────────────────────────────────────────────────────
        # Robust fist: all 4 fingers curled by angle + thumb not extended
        if three_curled and not thumb_up:
            return "FIST"

        # ── OPEN PALM ────────────────────────────────────────────────────────
        # All 4 fingers extended + spread (index↔pinky distance / hand_size)
        if n_four >= 4:
            spread = tracker.dist_norm(hand_lms, INDEX_TIP, PINKY_TIP) / scale
            if spread > self.SPREAD_THRESH:
                return "OPEN_PALM"

        return "NONE"


# ── Swipe detector ────────────────────────────────────────────────────────────

class SwipeDetector:
    """
    Tracks index-tip x over a sliding window.
    Fires only when velocity is consistent for SWIPE_CONFIRM_FRAMES frames
    and resets the buffer after firing to prevent double-triggers.
    """

    HISTORY = 8

    def __init__(self):
        self._xs  = deque(maxlen=self.HISTORY)
        self._buf = deque(maxlen=SWIPE_CONFIRM_FRAMES)

    def update(self, hand_lms) -> str:
        self._xs.append(hand_lms.landmark[INDEX_TIP].x)
        if len(self._xs) < 5:
            return "NONE"

        # Velocity = displacement over last 5 samples (smoothed over middle range)
        dx = float(self._xs[-1]) - float(self._xs[-5])

        if dx > SWIPE_VELOCITY_THRESHOLD:
            self._buf.append("R")
        elif dx < -SWIPE_VELOCITY_THRESHOLD:
            self._buf.append("L")
        else:
            self._buf.append("N")
            return "NONE"

        if len(self._buf) == SWIPE_CONFIRM_FRAMES:
            buf = list(self._buf)
            self._buf.clear()
            self._xs.clear()
            if all(d == "R" for d in buf):
                return "SWIPE_RIGHT"
            if all(d == "L" for d in buf):
                return "SWIPE_LEFT"

        return "NONE"


# ── Main mode class ───────────────────────────────────────────────────────────

class MusicPlayerMode:

    def __init__(self):
        self._clf        = GestureClassifier()
        self._swipe      = SwipeDetector()
        self._last_ts    = 0.0
        self._last_g     = ""
        self._disp       = ""
        self._disp_ts    = 0.0

    def process(self, frame: np.ndarray, results, tracker: HandTracker) -> np.ndarray:
        gesture = "NONE"
        if results.multi_hand_landmarks:
            hand_lms   = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            swipe = self._swipe.update(hand_lms)
            if swipe != "NONE":
                gesture = swipe
            else:
                gesture = self._clf.update(hand_lms, handedness, tracker)

            self._fire(gesture)

        self._draw_ui(frame, gesture)
        return frame

    def _fire(self, gesture: str):
        now      = time.time()
        cooldown = (now - self._last_ts) >= MUSIC_ACTION_COOLDOWN_SEC
        is_swipe = gesture.startswith("SWIPE_")
        changed  = gesture != self._last_g

        self._last_g = gesture
        if gesture == "NONE" or not cooldown:
            return
        if not is_swipe and not changed:
            return

        actions = {
            "OPEN_PALM":   (PLAY_PAUSE_KEY, "PLAY / PAUSE"),
            "FIST":        (PLAY_PAUSE_KEY, "PLAY / PAUSE"),
            "SWIPE_RIGHT": (NEXT_KEY,       "NEXT  >>"),
            "SWIPE_LEFT":  (PREV_KEY,       "<<  PREV"),
            "THUMBS_UP":   (VOL_UP_KEY,     "VOL +"),
            "THUMBS_DOWN": (VOL_DOWN_KEY,   "VOL -"),
        }
        if gesture in actions:
            key, label = actions[gesture]
            _send_media_key(key)
            self._disp    = f"{gesture}  ->  {label}"
            self._disp_ts = now
            self._last_ts = now

    def _draw_ui(self, frame: np.ndarray, gesture: str):
        h, w = frame.shape[:2]

        # Gesture indicator panel
        cv2.rectangle(frame, (0, h - 90), (480, h), (0, 0, 0), cv2.FILLED)
        colour = (0, 255, 180) if gesture != "NONE" else (80, 80, 80)
        cv2.putText(frame, f"Gesture: {gesture}",
                    (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        if time.time() - self._disp_ts < 2.5:
            cv2.putText(frame, self._disp,
                        (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 220, 255), 2)

        # Legend
        legend = [
            "Open Palm  ->  Play/Pause",
            "Fist       ->  Play/Pause",
            "Swipe -->  ->  Next track",
            "Swipe <--  ->  Prev track",
            "Thumbs Up  ->  Vol +",
            "Thumbs Dn  ->  Vol -",
        ]
        for i, txt in enumerate(legend):
            cv2.putText(frame, txt, (w - 310, 75 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1)
