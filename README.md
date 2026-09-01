# YOLOv8 Smart Surveillance Toolkit

Two standalone Python scripts that use **OpenCV** and **Ultralytics YOLOv8** to analyze video footage for common security/safety use cases:

1. **Crowd Density Analyzer** — counts people per frame and flags overcrowding.
2. **Abandoned Object Detector** — flags luggage items left unattended for too long.

Both scripts process a video frame-by-frame, show a live preview window while running, and save the annotated result to a new video file.

---

## Demo Preview

| Script | What it detects | Alert shown |
|---|---|---|
| `crowd_density_analyzer.py` | People (COCO class 0) | `OVERCROWDING DETECTED: N People` (red) / `Normal Density: N People` (green) |
| `abandoned_object_detector.py` | Backpacks, umbrellas, handbags, suitcases + people | `ALERT: SUSPECT ABANDONED OBJECT` (thick red box) |

---

## Requirements

- Python 3.9+
- PyCharm (or any IDE/terminal)

Install dependencies:

```bash
pip install ultralytics opencv-python
```

The first run will auto-download `yolov8n.pt` (the pretrained YOLOv8 "nano" model) if it isn't already present in the project folder.

---

## Project Structure

```
your-project/
├── crowd_density_analyzer.py
├── abandoned_object_detector.py
├── yolov8n.pt              # auto-downloaded on first run
├── input_video.mp4         # your source video (you provide this)
└── output_video.mp4        # generated after running a script
```

---

## Usage

1. Place a video named **`input_video.mp4`** in the same folder as the scripts.
2. Run either script from PyCharm or the terminal:

```bash
python crowd_density_analyzer.py
```

```bash
python abandoned_object_detector.py
```

3. A live preview window opens showing detections in real time. Press **`q`** at any time to stop processing early.
4. The fully processed video is saved as **`output_video.mp4`** in the same folder.

---

## How It Works

### 1. Crowd Density Analyzer
- Runs YOLOv8 inference on every frame, filtering for `person` (class ID `0`).
- Counts the number of person detections in the frame.
- If the count exceeds a configurable threshold (default: **10**), overlays a red "OVERCROWDING DETECTED" message; otherwise shows a green "Normal Density" message.

### 2. Abandoned Object Detector
- Runs YOLOv8 inference filtering for `person` (`0`) and luggage classes: `backpack` (`24`), `umbrella` (`25`), `handbag` (`26`), `suitcase` (`28`).
- Uses a lightweight, from-scratch **centroid tracker** to follow each luggage item across frames (nearest-centroid matching — no external tracking library required).
- If an item's position barely moves (within a pixel threshold) for more than **5 consecutive seconds** (calculated from the video's FPS) **and** no person is detected nearby, the item is flagged as abandoned with a thick red bounding box and warning text.

---

## Configuration

Both scripts expose tunable constants near the top of the file:

**`crowd_density_analyzer.py`**
```python
OVERCROWDING_THRESHOLD = 10   # people count that triggers the alert
```

**`abandoned_object_detector.py`**
```python
STATIONARY_PIXEL_THRESHOLD = 25     # max movement (px) still considered "stationary"
MATCH_DISTANCE_THRESHOLD = 75       # max distance (px) to match a detection to an existing track
PROXIMITY_DISTANCE_THRESHOLD = 150  # distance (px) within which a person "claims" an object
ABANDONED_SECONDS_THRESHOLD = 5     # seconds an object must be unattended to trigger an alert
```

---

## Notes & Limitations

- The abandoned-object tracker is intentionally simple (built for learning/demo purposes) — it uses nearest-centroid matching rather than a production-grade tracker like DeepSORT or ByteTrack, so heavy occlusion or crowded scenes may cause ID switches.
- Detection accuracy depends on the YOLOv8 model used — swap `yolov8n.pt` for `yolov8s.pt` / `yolov8m.pt` for higher accuracy at the cost of speed.
- Both scripts default to CPU inference; if you have a CUDA-capable GPU and the GPU build of PyTorch installed, Ultralytics will automatically use it for faster processing.

---

## License

Free to use and modify for learning, prototyping, or portfolio projects.
