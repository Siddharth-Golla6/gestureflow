# ─────────────────────────────────────────────
#  mode_virtual_mouse.py
# ─────────────────────────────────────────────

import time
import cv2
import numpy as np
import pyautogui
from collections import deque
from hand_tracker import HandTracker, THUMB_TIP, INDEX_TIP, MIDDLE_TIP
from config import (
    MOUSE_REGION_LEFT, MOUSE_REGION_RIGHT,
    MOUSE_REGION_TOP,  MOUSE_REGION_BOTTOM,
    MOUSE_SMOOTH_ALPHA, MOUSE_SMOOTH_WINDOW,
    MOUSE_DEAD_ZONE_PX,
    CLICK_LEFT_THRESHOLD, CLICK_RIGHT_THRESHOLD,
    SCROLL_PINCH_THRESHOLD, SCROLL_SPEED,
    CLICK_COOLDOWN_SEC,
)

pyautogui.FAILSAFE = False
_SW, _SH = pyautogui.size()

_CLICK_CONFIRM = 3   # frames pinch must be held before click fires


class _TwoStageSmoother:
    """
    Stage 1: sliding-window mean over the last N raw target positions.
             Kills high-frequency jitter from landmark noise.
    Stage 2: EMA on the stage-1 output.
             Gives momentum so the cursor glides rather than stutters.
    Dead zone: if the smoothed target moved less than DEAD_ZONE_PX pixels
               from the last sent position, don't call moveTo at all.
    """

    def __init__(self):
        self._xs  = deque(maxlen=MOUSE_SMOOTH_WINDOW)
        self._ys  = deque(maxlen=MOUSE_SMOOTH_WINDOW)
        self._cx  = float(_SW) / 2
        self._cy  = float(_SH) / 2
        self._sent_x = self._cx
        self._sent_y = self._cy

    def update(self, tx: float, ty: float):
        self._xs.append(tx)
        self._ys.append(ty)

        # Stage 1 — window mean
        mx = float(np.mean(self._xs))
        my = float(np.mean(self._ys))

        # Stage 2 — EMA
        a = MOUSE_SMOOTH_ALPHA
        self._cx = a * mx + (1 - a) * self._cx
        self._cy = a * my + (1 - a) * self._cy

        cx = int(np.clip(self._cx, 0, _SW - 1))
        cy = int(np.clip(self._cy, 0, _SH - 1))

        # Dead zone — skip tiny tremors
        if (abs(cx - self._sent_x) < MOUSE_DEAD_ZONE_PX and
                abs(cy - self._sent_y) < MOUSE_DEAD_ZONE_PX):
            return

        self._sent_x = cx
        self._sent_y = cy
        pyautogui.moveTo(cx, cy)

    @property
    def pos(self):
        return int(self._cx), int(self._cy)


class VirtualMouseMode:

    def __init__(self):
        self._smoother      = _TwoStageSmoother()
        self._click_ts      = 0.0
        self._scroll_anchor = None   # y position when scroll mode was entered
        self._scroll_accum  = 0.0   # accumulated vertical movement
        self._status        = ""
        self._status_ts     = 0.0
        self._lbuf          = deque(maxlen=_CLICK_CONFIRM)
        self._rbuf          = deque(maxlen=_CLICK_CONFIRM)

    def process(self, frame: np.ndarray, results, tracker: HandTracker) -> np.ndarray:
        h, w = frame.shape[:2]
        self._draw_region(frame, w, h)

        if results.multi_hand_landmarks:
            hand_lms   = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            scale    = max(tracker.hand_size(hand_lms), 1e-4)
            d_li     = tracker.dist_norm(hand_lms, THUMB_TIP, INDEX_TIP)  / scale
            d_lm     = tracker.dist_norm(hand_lms, THUMB_TIP, MIDDLE_TIP) / scale
            d_im     = tracker.dist_norm(hand_lms, INDEX_TIP, MIDDLE_TIP) / scale

            fingers      = tracker.fingers_up(hand_lms, handedness)
            scroll_mode  = fingers[1] and fingers[2] and d_im < SCROLL_PINCH_THRESHOLD
            now          = time.time()
            cooldown_ok  = (now - self._click_ts) >= CLICK_COOLDOWN_SEC

            if scroll_mode:
                self._handle_scroll(hand_lms)
                self._lbuf.clear()
                self._rbuf.clear()
            else:
                # Reset scroll state when leaving scroll mode
                self._scroll_anchor = None
                self._scroll_accum  = 0.0

                # Move only when index is up and not pinching
                if fingers[1] and d_li > CLICK_LEFT_THRESHOLD:
                    ix = hand_lms.landmark[INDEX_TIP].x
                    iy = hand_lms.landmark[INDEX_TIP].y
                    tx, ty = self._to_screen(ix, iy)
                    self._smoother.update(tx, ty)

                # Click confirmation buffers
                self._lbuf.append(d_li < CLICK_LEFT_THRESHOLD)
                self._rbuf.append(d_lm < CLICK_RIGHT_THRESHOLD and
                                  d_li > CLICK_LEFT_THRESHOLD)

                if all(self._lbuf) and cooldown_ok:
                    pyautogui.click()
                    self._set_status("LEFT CLICK")
                    self._click_ts = now
                    self._lbuf.clear()
                elif all(self._rbuf) and cooldown_ok:
                    pyautogui.rightClick()
                    self._set_status("RIGHT CLICK")
                    self._click_ts = now
                    self._rbuf.clear()

            # Fingertip dot
            fx, fy   = tracker.landmark_px(hand_lms, INDEX_TIP, w, h)
            clicking = d_li < CLICK_LEFT_THRESHOLD
            cv2.circle(frame, (fx, fy), 13, (0, 60, 255) if clicking else (0, 255, 100), cv2.FILLED)
            cv2.circle(frame, (fx, fy), 13, (255, 255, 255), 1)
            cv2.putText(frame, f"d={d_li:.2f}", (fx + 15, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 1)

        self._draw_ui(frame)
        return frame

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_screen(nx: float, ny: float):
        """Map normalised hand position inside active region → screen pixels."""
        rx = max(MOUSE_REGION_LEFT,  min(MOUSE_REGION_RIGHT,  nx))
        ry = max(MOUSE_REGION_TOP,   min(MOUSE_REGION_BOTTOM, ny))
        rw = MOUSE_REGION_RIGHT  - MOUSE_REGION_LEFT
        rh = MOUSE_REGION_BOTTOM - MOUSE_REGION_TOP
        return (rx - MOUSE_REGION_LEFT) / rw * _SW, \
               (ry - MOUSE_REGION_TOP)  / rh * _SH

    def _handle_scroll(self, hand_lms):
        # Use midpoint of index+middle tips for a stable scroll anchor
        iy = (hand_lms.landmark[INDEX_TIP].y + hand_lms.landmark[MIDDLE_TIP].y) / 2.0

        if self._scroll_anchor is None:
            self._scroll_anchor = iy
            return

        # Accumulate movement from the fixed anchor
        delta = iy - self._scroll_anchor          # positive = hand moved down
        self._scroll_accum += delta
        self._scroll_anchor = iy                  # slide anchor each frame

        # One scroll step = 0.02 normalised units of hand movement
        STEP = 0.018
        ticks = int(self._scroll_accum / STEP)
        if ticks != 0:
            pyautogui.scroll(-ticks * 3)          # negative = scroll down
            self._scroll_accum -= ticks * STEP
            direction = "DOWN" if ticks > 0 else "UP"
            self._set_status(f"SCROLL {direction}")

    def _set_status(self, text: str):
        self._status    = text
        self._status_ts = time.time()

    def _draw_region(self, frame, w, h):
        rx1 = int(MOUSE_REGION_LEFT   * w)
        rx2 = int(MOUSE_REGION_RIGHT  * w)
        ry1 = int(MOUSE_REGION_TOP    * h)
        ry2 = int(MOUSE_REGION_BOTTOM * h)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (50, 200, 255), 2)
        cv2.putText(frame, "Move hand here", (rx1 + 8, ry1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 255), 1)

    def _draw_ui(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if time.time() - self._status_ts < 1.5:
            cv2.putText(frame, self._status, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 3)
        cx, cy = self._smoother.pos
        cv2.putText(frame, f"Cursor ({cx}, {cy})",
                    (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        tips = [
            "Index up = move cursor",
            "Thumb+Index pinch = Left click",
            "Thumb+Middle pinch = Right click",
            "Index+Middle close = Scroll",
            "SPACE = freeze",
        ]
        for i, t in enumerate(tips):
            cv2.putText(frame, t, (w - 320, 80 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
