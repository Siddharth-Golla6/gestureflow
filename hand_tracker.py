# ─────────────────────────────────────────────
#  hand_tracker.py  —  Shared MediaPipe wrapper
# ─────────────────────────────────────────────

import cv2
import mediapipe as mp
import numpy as np
from config import (
    MEDIAPIPE_MAX_HANDS,
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_TRACKING_CONFIDENCE,
)

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST       = 0
THUMB_CMC   = 1; THUMB_MCP  = 2; THUMB_IP   = 3; THUMB_TIP  = 4
INDEX_MCP   = 5; INDEX_PIP  = 6; INDEX_DIP  = 7; INDEX_TIP  = 8
MIDDLE_MCP  = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP  = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_PIP = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def _vec2(lms, a, b):
    """2-D vector from landmark a → b."""
    return np.array([lms[b].x - lms[a].x, lms[b].y - lms[a].y], dtype=np.float32)


def _angle_deg(v1, v2):
    """Angle in degrees between two 2-D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


class HandTracker:
    """MediaPipe Hands wrapper — initialised once, reused across all modes."""

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_style = mp.solutions.drawing_styles
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MEDIAPIPE_MAX_HANDS,
            min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        return results

    def draw_landmarks(self, frame_bgr, results):
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame_bgr, hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_style.get_default_hand_landmarks_style(),
                    self._mp_style.get_default_hand_connections_style(),
                )
        return frame_bgr

    # ── Distance helpers ─────────────────────────────────────────────────────

    @staticmethod
    def landmark_px(hand_lms, idx, w, h):
        lm = hand_lms.landmark[idx]
        return int(lm.x * w), int(lm.y * h)

    @staticmethod
    def landmark_norm(hand_lms, idx):
        lm = hand_lms.landmark[idx]
        return lm.x, lm.y

    @staticmethod
    def dist_norm(hand_lms, a, b):
        """Raw Euclidean distance between two landmarks (normalised 0-1 coords)."""
        lms = hand_lms.landmark
        return float(np.hypot(lms[a].x - lms[b].x, lms[a].y - lms[b].y))

    @staticmethod
    def hand_size(hand_lms):
        """Wrist → middle-MCP distance; use to normalise other distances."""
        return HandTracker.dist_norm(hand_lms, WRIST, MIDDLE_MCP)

    @staticmethod
    def dist_scaled(hand_lms, a, b):
        """Distance between a and b divided by hand_size (scale-invariant)."""
        sz = max(HandTracker.hand_size(hand_lms), 1e-4)
        return HandTracker.dist_norm(hand_lms, a, b) / sz

    # ── Angle-based finger extension ─────────────────────────────────────────
    # For each of the 4 fingers we compute the angle at the PIP joint:
    #   vector MCP→PIP  and  vector PIP→TIP
    # If angle > EXTEND_THRESH the finger is straight (extended).
    # If angle < CURL_THRESH the finger is clearly curled.
    # This is ROTATION-INVARIANT — works regardless of hand tilt.

    EXTEND_THRESH = 160   # degrees — finger considered fully extended
    CURL_THRESH   = 100   # degrees — finger considered fully curled

    @staticmethod
    def finger_angle(hand_lms, mcp, pip, tip):
        """Angle at PIP joint in degrees (180 = fully straight)."""
        lms = hand_lms.landmark
        v_back   = _vec2(lms, pip, mcp)   # PIP → MCP (back toward palm)
        v_forward = _vec2(lms, pip, tip)  # PIP → TIP (forward)
        return _angle_deg(v_back, v_forward)

    @staticmethod
    def thumb_angle(hand_lms):
        """Angle at THUMB_MCP joint (CMC→MCP→TIP)."""
        lms = hand_lms.landmark
        v_back    = _vec2(lms, THUMB_MCP, THUMB_CMC)
        v_forward = _vec2(lms, THUMB_MCP, THUMB_TIP)
        return _angle_deg(v_back, v_forward)

    @classmethod
    def fingers_up(cls, hand_lms, handedness_label="Right"):
        """
        Return [thumb, index, middle, ring, pinky] booleans.

        Four fingers: angle at PIP > EXTEND_THRESH  →  extended.
        Thumb: angle at MCP  > 130°  AND
               thumb-tip farther from index-MCP than thumb-MCP is
               (catches both upward and sideways thumb extension).
        """
        lms = hand_lms.landmark
        up  = []

        # Thumb (two-condition to avoid false positives)
        t_angle = cls.thumb_angle(hand_lms)
        d_tip   = np.hypot(lms[THUMB_TIP].x - lms[INDEX_MCP].x,
                           lms[THUMB_TIP].y - lms[INDEX_MCP].y)
        d_mcp   = np.hypot(lms[THUMB_MCP].x - lms[INDEX_MCP].x,
                           lms[THUMB_MCP].y - lms[INDEX_MCP].y)
        up.append(t_angle > 130 and d_tip > d_mcp * 1.15)

        # Index
        up.append(cls.finger_angle(hand_lms, INDEX_MCP,  INDEX_PIP,  INDEX_TIP)  > cls.EXTEND_THRESH)
        # Middle
        up.append(cls.finger_angle(hand_lms, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP) > cls.EXTEND_THRESH)
        # Ring
        up.append(cls.finger_angle(hand_lms, RING_MCP,   RING_PIP,   RING_TIP)   > cls.EXTEND_THRESH)
        # Pinky
        up.append(cls.finger_angle(hand_lms, PINKY_MCP,  PINKY_PIP,  PINKY_TIP)  > cls.EXTEND_THRESH)

        return up

    @classmethod
    def finger_curled(cls, hand_lms, mcp, pip, tip):
        """True if this finger is clearly curled (angle at PIP < CURL_THRESH)."""
        return cls.finger_angle(hand_lms, mcp, pip, tip) < cls.CURL_THRESH

    @staticmethod
    def palm_center_norm(hand_lms):
        idxs = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        xs = [hand_lms.landmark[i].x for i in idxs]
        ys = [hand_lms.landmark[i].y for i in idxs]
        return float(np.mean(xs)), float(np.mean(ys))

    def close(self):
        self.hands.close()
