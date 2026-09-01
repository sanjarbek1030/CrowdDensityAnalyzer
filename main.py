"""
Crowd Density Analyzer
------------------------
This script reads 'input_video.mp4', uses a YOLOv8 model to detect people
in every frame, counts them, and overlays a warning message if the crowd
count exceeds a set threshold. The processed video is displayed live in a
window and saved to 'output_video.mp4'.

Requirements (install these first in your PyCharm terminal):
    pip install ultralytics opencv-python

Make sure 'yolov8n.pt' and 'input_video.mp4' are in the same folder as
this script (or update the paths below).
"""

import cv2                     # OpenCV: used for reading/writing video and drawing overlays
from ultralytics import YOLO   # Ultralytics YOLOv8: used for object detection

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_VIDEO_PATH = "input_video.mp4"     # Path to the video we want to analyze
OUTPUT_VIDEO_PATH = "output_video.mp4"   # Path where the processed video will be saved
MODEL_PATH = "yolov8n.pt"                # Pretrained YOLOv8 "nano" model (small & fast)
PERSON_CLASS_ID = 0                      # In the COCO dataset (which YOLOv8n is trained on), class 0 = "person"
OVERCROWDING_THRESHOLD = 10              # If more than this many people are detected, flag "overcrowding"

# ---------------------------------------------------------------------------
# STEP 1: Load the YOLOv8 model
# ---------------------------------------------------------------------------
try:
    # This loads the pretrained YOLOv8 nano weights from disk.
    # If the file doesn't exist locally, Ultralytics will try to download it automatically.
    model = YOLO(MODEL_PATH)
except Exception as e:
    # If the model fails to load (bad path, corrupted file, no internet, etc.), stop the script cleanly.
    print(f"[ERROR] Could not load YOLO model '{MODEL_PATH}': {e}")
    exit()  # Exit the program since we cannot continue without a model

# ---------------------------------------------------------------------------
# STEP 2: Open the input video file
# ---------------------------------------------------------------------------
try:
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)  # Create a VideoCapture object to read frames from the file

    # If OpenCV could not open the file (wrong path, unsupported codec, etc.), it returns False here.
    if not cap.isOpened():
        raise IOError(f"Cannot open video file '{INPUT_VIDEO_PATH}'")
except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# ---------------------------------------------------------------------------
# STEP 3: Read video properties (needed to set up the output writer correctly)
# ---------------------------------------------------------------------------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # Width of each frame in pixels
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of each frame in pixels
fps = cap.get(cv2.CAP_PROP_FPS)                         # Frames per second of the input video

# Some video files report fps as 0 due to metadata issues; default to 30 fps to avoid a broken writer.
if fps <= 0:
    fps = 30.0

# ---------------------------------------------------------------------------
# STEP 4: Set up the VideoWriter to save the processed output
# ---------------------------------------------------------------------------
# fourcc is a 4-character code that specifies the video codec used to compress the frames.
# 'mp4v' is a widely compatible codec for saving .mp4 files.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create the VideoWriter object: output path, codec, fps, and frame size (width, height) must match the input.
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print("[INFO] Starting video processing... Press 'q' in the display window to stop early.")

# ---------------------------------------------------------------------------
# STEP 5: Main frame-by-frame processing loop
# ---------------------------------------------------------------------------
while True:
    ret, frame = cap.read()  # Read the next frame; 'ret' is True if a frame was successfully read

    if not ret:
        # 'ret' is False when the video has ended (no more frames left) or an error occurred.
        print("[INFO] End of video reached (or cannot read further frames).")
        break

    # -----------------------------------------------------------------------
    # Run YOLOv8 detection on this single frame.
    # classes=[PERSON_CLASS_ID] tells YOLO to only return detections for "person".
    # verbose=False suppresses YOLO's per-frame console logging to keep output clean.
    # -----------------------------------------------------------------------
    results = model.predict(source=frame, classes=[PERSON_CLASS_ID], verbose=False)

    # results is a list (one entry per image); since we passed one frame, we take the first result.
    result = results[0]

    # result.boxes contains all detected bounding boxes for this frame.
    person_count = len(result.boxes)  # The number of detected "person" boxes = number of people in the frame

    # -----------------------------------------------------------------------
    # Draw a bounding box around every detected person so the user can visually verify the count.
    # -----------------------------------------------------------------------
    for box in result.boxes:
        # box.xyxy holds the coordinates [x1, y1, x2, y2] of the bounding box corners.
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Draw a simple blue rectangle (BGR color format) around each detected person.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # -----------------------------------------------------------------------
    # Decide which message and color to display based on the crowd threshold.
    # -----------------------------------------------------------------------
    if person_count > OVERCROWDING_THRESHOLD:
        message = f"OVERCROWDING DETECTED: {person_count} People"
        text_color = (0, 0, 255)   # Bright red in BGR format (Blue=0, Green=0, Red=255)
    else:
        message = f"Normal Density: {person_count} People"
        text_color = (0, 255, 0)   # Green in BGR format

    # -----------------------------------------------------------------------
    # Overlay the message text onto the top-left corner of the frame.
    # -----------------------------------------------------------------------
    cv2.putText(
        frame,                     # The image/frame to draw on
        message,                   # The text string to display
        (20, 40),                  # Bottom-left corner coordinates of the text (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font style
        1.0,                       # Font scale (size)
        text_color,                # Text color (BGR)
        2,                         # Text thickness
        cv2.LINE_AA                # Anti-aliased line type for smoother-looking text
    )

    # -----------------------------------------------------------------------
    # Write the processed frame (with boxes and text) to the output video file.
    # -----------------------------------------------------------------------
    out.write(frame)

    # -----------------------------------------------------------------------
    # Display the processed frame live in a window so the user can watch progress.
    # -----------------------------------------------------------------------
    cv2.imshow("Crowd Density Analyzer", frame)

    # Wait 1 millisecond for a key press; if the user presses 'q', stop processing early.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] 'q' pressed. Stopping early.")
        break

# ---------------------------------------------------------------------------
# STEP 6: Clean up — release resources so files are properly saved and closed
# ---------------------------------------------------------------------------
cap.release()           # Release the input video file
out.release()            # Finalize and close the output video file (IMPORTANT: without this, the file may be corrupted)
cv2.destroyAllWindows()  # Close any OpenCV display windows

print(f"[INFO] Processing complete. Output saved to '{OUTPUT_VIDEO_PATH}'.")
