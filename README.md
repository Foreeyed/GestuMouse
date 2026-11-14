# GestuMouse

Control the mouse cursor with hand gestures using a webcam.

## How it works

1. **MediaPipe Hands** finds 21 landmarks of the hand.
2. A **RandomForest classifier** recognizes the gesture from normalized landmarks.
3. The cursor is moved to the index finger tip with moving-average smoothing.

## Gestures

| Gesture | Action |
| --- | --- |
| ☝️ `pointing` | move cursor |
| ✊ `fist` | single click |
| 🖐 `open_hand` | idle |
| 👍 `thumbs_up` | double click |

A gesture must be held for 5 consecutive frames before it activates (debounce).
Clicks fire only when the gesture activates, not on every frame.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### macOS permissions

The app needs **Accessibility** permission to move the mouse:
System Settings → Privacy & Security → Accessibility → enable your terminal / IDE.
The camera prompt must also be allowed.

## Usage

### 1. Collect gesture data (optional, dataset already included)

```bash
python collect_data.py
```

- `0`..`3` — switch the gesture being recorded
- `R` — start/stop recording (3 samples per second)
- `Q` — quit and save to `gestures_dataset.csv`

> The included dataset contains only `fist` and `open_hand`.
> Collect `pointing` and `thumbs_up` data to enable the full gesture scheme.

### 2. Train the model

```bash
python train.py
```

Prints a classification report and cross-validation accuracy, then saves
`model.pkl`.

### 3. Run GestuMouse

```bash
python main.py
```

Press `q` in the camera window to quit.

## Project structure

```
main.py            # main loop: camera -> tracking -> gestures -> mouse
tracking.py        # MediaPipe Hands wrapper
gestures.py        # feature extraction + model prediction
mouse_control.py   # cursor movement, smoothing, clicks
train.py           # trains the model and saves model.pkl
collect_data.py    # records gesture samples into gestures_dataset.csv
gestures_dataset.csv
model.pkl
```
