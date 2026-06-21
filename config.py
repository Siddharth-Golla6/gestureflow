# ─────────────────────────────────────────────
#  config.py  —  All tunable thresholds
# ─────────────────────────────────────────────

# ── General ──────────────────────────────────
CAMERA_INDEX = 1
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
FPS_TARGET   = 30

# Hotkeys
KEY_MODE_VOLUME = ord('1')
KEY_MODE_MUSIC  = ord('2')
KEY_MODE_MOUSE  = ord('3')
KEY_PAUSE       = ord(' ')   # spacebar freezes all gesture control
KEY_VOLUME_SUB  = ord('v')   # toggle volume sub-mode (volume vs brightness)
KEY_BRIGHTNESS  = ord('b')

# ── Hand Tracking ─────────────────────────────
MEDIAPIPE_MAX_HANDS            = 2
MEDIAPIPE_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_TRACKING_CONFIDENCE  = 0.6

# ── Mode 1: Volume / Brightness ───────────────
# Pinch distance DIVIDED BY hand_size (scale-invariant)
VOL_BRIGHT_MIN_DIST = 0.20   # fingers fully together
VOL_BRIGHT_MAX_DIST = 1.20   # fingers fully spread

SMOOTHING_WINDOW      = 8    # frames for moving-average (larger = smoother Vol/Bri)
BRIGHTNESS_MIN_CHANGE = 2    # only call set_brightness when change exceeds this %

# ── Mode 2: Music Player ──────────────────────
# Gesture cooldown so the same action isn't fired every frame
MUSIC_ACTION_COOLDOWN_SEC = 1.0

# Swipe: minimum horizontal displacement of index tip over 5 frames
SWIPE_VELOCITY_THRESHOLD = 0.030
# Consecutive frames the swipe direction must be consistent before firing
SWIPE_CONFIRM_FRAMES = 4

# ── Mode 3: Virtual Mouse ─────────────────────
MOUSE_REGION_LEFT   = 0.20
MOUSE_REGION_RIGHT  = 0.80
MOUSE_REGION_TOP    = 0.15
MOUSE_REGION_BOTTOM = 0.85

# EMA alpha for cursor (lower = smoother but more lag, higher = responsive but jittery)
MOUSE_SMOOTH_ALPHA = 0.12

# Secondary smoothing: average over this many recent target positions
MOUSE_SMOOTH_WINDOW = 6

# Dead-zone radius in pixels — tiny shakes below this are ignored
MOUSE_DEAD_ZONE_PX = 6

# Click pinch thresholds — distances DIVIDED BY hand_size (scale-invariant)
# ~0.3 = touching, ~1.0 = wide open
CLICK_LEFT_THRESHOLD   = 0.35   # thumb ↔ index
CLICK_RIGHT_THRESHOLD  = 0.38   # thumb ↔ middle
SCROLL_PINCH_THRESHOLD = 0.40   # index ↔ middle (scroll mode)
SCROLL_SPEED = 15               # pixels per scroll tick

# Cooldown between clicks (seconds)
CLICK_COOLDOWN_SEC = 0.4

# ── Thumbs-up gesture (mode cycle) ───────────
THUMBSUP_HOLD_SEC = 1.0         # hold thumbs-up this long to cycle mode
