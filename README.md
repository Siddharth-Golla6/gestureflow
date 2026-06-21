# GestureFlow

Control your PC hands-free using only your webcam and hand gestures.  
Switch between three modes — Volume/Brightness, Music Player, and Virtual Mouse — all powered by a single MediaPipe hand-tracking pipeline.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## Demo

| Mode | Gesture | Action |
|------|---------|--------|
| Volume/Brightness | Pinch thumb ↔ index | Adjust level |
| Music Player | Open palm | Play / Pause |
| Music Player | Swipe right / left | Next / Prev track |
| Music Player | Thumbs up / down | Volume +/- |
| Virtual Mouse | Move index finger | Move cursor |
| Virtual Mouse | Thumb + index pinch | Left click |
| Virtual Mouse | Thumb + middle pinch | Right click |
| Virtual Mouse | Index + middle close | Scroll |

---

## Features

- **Mode 1 — Volume & Brightness:** Pinch distance controls system volume or screen brightness. Two hands tracked simultaneously (left = volume, right = brightness). Live bar overlay updates in real time with 8-frame smoothing.
- **Mode 2 — Music Player:** Gesture recognition sends media keys to whatever app is focused (Spotify, YouTube, VLC, etc.). Angle-based finger detection works at any hand tilt. 6-frame majority voting prevents false triggers.
- **Mode 3 — Virtual Mouse:** Index fingertip maps to full screen via two-stage smoothing (window average + EMA). Click confirmation requires holding a pinch for 3 frames. Accumulator-based scroll for smooth, consistent scrolling.
- **Always on:** MediaPipe 21-point skeleton overlay on live feed, FPS counter, mode indicator, pause hotkey.

---

## Project Structure

```
gestureflow/
├── main.py                   # Camera loop, mode switching, HUD
├── hand_tracker.py           # MediaPipe wrapper (shared, initialised once)
├── mode_volume_brightness.py # Mode 1
├── mode_music_player.py      # Mode 2
├── mode_virtual_mouse.py     # Mode 3
├── config.py                 # All tunable thresholds
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9 or higher
- Webcam (built-in or external; Camo virtual camera supported)

---

## Installation

```bash
git clone https://github.com/WATERFBG/gestureflow.git
cd gestureflow
pip install -r requirements.txt
```

### OS-specific notes

#### Windows
`pycaw` handles system volume automatically. No extra steps needed.

#### macOS
`pycaw` is Windows-only. Comment it out from `requirements.txt`.  
Volume is controlled via `osascript` (built-in — no install needed).  
Grant Accessibility permission: *System Settings → Privacy & Security → Accessibility → add Terminal/Python*

#### Linux
```bash
sudo apt install alsa-utils      # for amixer (volume)
sudo usermod -aG video $USER     # for screen brightness (logout to apply)
pip install python3-xlib         # for pyautogui
```

---

## Running

```bash
python main.py
```

A window opens showing your live camera feed with hand skeleton overlay.

---

## Controls

| Key | Action |
|-----|--------|
| `1` | Switch to Volume / Brightness mode |
| `2` | Switch to Music Player mode |
| `3` | Switch to Virtual Mouse mode |
| `Space` | **Freeze / unfreeze** all gesture control |
| `v` | (Mode 1) Single hand controls Volume |
| `b` | (Mode 1) Single hand controls Brightness |
| `Q` / `Esc` | Quit |

> **Gesture shortcut:** Hold a thumbs-up for 1 second to cycle through modes.

---

## Gesture Reference

### Mode 1 — Volume & Brightness

Pinch thumb and index finger together or spread them apart.

- Close pinch → low level
- Wide spread → high level
- **One hand:** press `v` (volume) or `b` (brightness) to choose
- **Two hands:** left hand = volume, right hand = brightness

### Mode 2 — Music Player

| Gesture | Action |
|---------|--------|
| Open palm (fingers spread wide) | Play / Pause |
| Closed fist | Play / Pause |
| Swipe hand right | Next track |
| Swipe hand left | Previous track |
| Thumbs up | Volume up |
| Thumbs down | Volume down |

### Mode 3 — Virtual Mouse

Move your hand **inside the blue Active Region** shown on screen. The region maps to your full screen.

| Gesture | Action |
|---------|--------|
| Index fingertip movement | Move cursor |
| Thumb + index pinch (hold 3 frames) | Left click |
| Thumb + middle pinch (hold 3 frames) | Right click |
| Index + middle fingers close, move up/down | Scroll |

> Press `Space` immediately if the cursor moves unexpectedly — it freezes all gesture control.

---

## Configuration

All thresholds live in `config.py`. Key values to tune per your hand size and camera distance:

```python
# Camera
CAMERA_INDEX = 0          # change if wrong camera opens

# Mouse smoothing (lower = smoother, higher = more responsive)
MOUSE_SMOOTH_ALPHA  = 0.12
MOUSE_SMOOTH_WINDOW = 6
MOUSE_DEAD_ZONE_PX  = 6

# Click sensitivity (scale-invariant: distance / hand_size)
CLICK_LEFT_THRESHOLD  = 0.35
CLICK_RIGHT_THRESHOLD = 0.38

# Volume/Brightness pinch range
VOL_BRIGHT_MIN_DIST = 0.20
VOL_BRIGHT_MAX_DIST = 1.20

# Swipe speed
SWIPE_VELOCITY_THRESHOLD = 0.030
```

---

## Using an External Camera (Camo)

[Camo](https://reincubate.com/camo/) lets you use your iPhone as a webcam.

1. Install Camo on your PC and iPhone
2. Connect via USB
3. Confirm the live feed appears in the Camo desktop app
4. Run `python main.py` — GestureFlow auto-detects Camo

If it doesn't pick it up automatically, find the index:

```bash
python -c "
import cv2
for i in range(6):
    c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, f = c.read()
    print(f'Index {i}: opened={c.isOpened()}, brightness={f.sum() if ok and f is not None else 0}')
    c.release()
"
```

Set `CAMERA_INDEX` in `config.py` to whichever index has a non-zero brightness value.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Black camera feed | Run the index scanner above; set `CAMERA_INDEX` in `config.py` |
| Low FPS | Set `FRAME_WIDTH = 640`, `FRAME_HEIGHT = 480` in `config.py` |
| Volume not changing (Windows) | Run as Administrator; `pip install comtypes pycaw` |
| Cursor jittery | Lower `MOUSE_SMOOTH_ALPHA` (try `0.08`) |
| False clicks | Raise `CLICK_LEFT_THRESHOLD` (try `0.45`) |
| Gestures not detected | Ensure good lighting; keep hand inside frame |
| MediaPipe + TensorFlow conflict | `pip uninstall tensorflow && pip install mediapipe==0.10.13` |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| [MediaPipe](https://mediapipe.dev/) | Hand landmark detection (21 points) |
| [OpenCV](https://opencv.org/) | Camera capture, frame rendering |
| [pycaw](https://github.com/AndreMiras/pycaw) | Windows system volume control |
| [screen-brightness-control](https://github.com/Crozzers/screen-brightness-control) | Cross-platform brightness |
| [pyautogui](https://pyautogui.readthedocs.io/) | Mouse control, media keys |
| [NumPy](https://numpy.org/) | Landmark math, smoothing |

---

## License

MIT
