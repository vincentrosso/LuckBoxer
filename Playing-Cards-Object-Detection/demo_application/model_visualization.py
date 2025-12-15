# Base code provided by Dipankar Medhi article https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
# Note press Q to stop the demo

import math
import sys
from ultralytics import YOLO
import cv2

# Change to 'tuned' to use it as the default one
DEFAULT_MODEL = "synthetic"
SHOW_CONFIDENCE = False

import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
print(f"Running from {project_root}")

configuration_dict = {
    "synthetic": {
        "model_path": str(project_root / "final_models" / "yolov8m_synthetic.pt"),
        "class_names": [
            "10c",
            "10d",
            "10h",
            "10s",
            "2c",
            "2d",
            "2h",
            "2s",
            "3c",
            "3d",
            "3h",
            "3s",
            "4c",
            "4d",
            "4h",
            "4s",
            "5c",
            "5d",
            "5h",
            "5s",
            "6c",
            "6d",
            "6h",
            "6s",
            "7c",
            "7d",
            "7h",
            "7s",
            "8c",
            "8d",
            "8h",
            "8s",
            "9c",
            "9d",
            "9h",
            "9s",
            "Ac",
            "Ad",
            "Ah",
            "As",
            "Jc",
            "Jd",
            "Jh",
            "Js",
            "Kc",
            "Kd",
            "Kh",
            "Ks",
            "Qc",
            "Qd",
            "Qh",
            "Qs",
        ],
    },
    "tuned": {
        "model_path": str(project_root / "final_models" / "yolov8m_tuned.pt"),
        "class_names": ["10h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "Ah", "Jh", "Kh", "Qh"],
    },
}

print("Loading application...")

parser = argparse.ArgumentParser(description="Real-time playing card detection demo.")
parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="Model preset (synthetic|tuned)")
parser.add_argument(
    "--source",
    default="0",
    help="Video source: camera index (e.g. 0) or path to a video file.",
)
args = parser.parse_args()

configuration_model = args.model

if configuration_model not in configuration_dict.keys():
    print(f"Allowed parameters for model are {configuration_dict.keys()}. Defaulting to {DEFAULT_MODEL}...")
    configuration_model = DEFAULT_MODEL

current_config = configuration_dict.get(configuration_model)

# Load the model and class names
model = YOLO(current_config["model_path"])
classNames = current_config["class_names"]

def _parse_source(value: str):
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def _open_capture(source):
    if isinstance(source, int) and sys.platform == "darwin":
        return cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(source)


source = _parse_source(args.source)
cap = _open_capture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    msg = f"Could not open video source={args.source!r}."
    if sys.platform == "darwin":
        msg += (
            "\nOn macOS, grant Camera permission to the app running Python "
            "(e.g. Visual Studio Code / Terminal) in System Settings > Privacy & Security > Camera, "
            "then restart and try again."
        )
    raise SystemExit(msg)


# Card values mapping
card_values = {
    "2c": 2,
    "2d": 2,
    "2h": 2,
    "2s": 2,
    "3c": 3,
    "3d": 3,
    "3h": 3,
    "3s": 3,
    "4c": 4,
    "4d": 4,
    "4h": 4,
    "4s": 4,
    "5c": 5,
    "5d": 5,
    "5h": 5,
    "5s": 5,
    "6c": 6,
    "6d": 6,
    "6h": 6,
    "6s": 6,
    "7c": 7,
    "7d": 7,
    "7h": 7,
    "7s": 7,
    "8c": 8,
    "8d": 8,
    "8h": 8,
    "8s": 8,
    "9c": 9,
    "9d": 9,
    "9h": 9,
    "9s": 9,
    "10c": 10,
    "10d": 10,
    "10h": 10,
    "10s": 10,
    "Ac": 1,
    "Ad": 1,
    "Ah": 1,
    "As": 1,
    "Jc": 10,
    "Jd": 10,
    "Jh": 10,
    "Js": 10,
    "Kc": 10,
    "Kd": 10,
    "Kh": 10,
    "Ks": 10,
    "Qc": 10,
    "Qd": 10,
    "Qh": 10,
    "Qs": 10,
}

window_title = f"Playing Cards Detection - Model: {configuration_model}"

try:
    consecutive_failures = 0
    while True:
        success, img = cap.read()
        if not success or img is None or getattr(img, "size", 0) == 0:
            consecutive_failures += 1
            if consecutive_failures == 1:
                print(
                    "Failed to read a frame from the video source. "
                    "If you're using a camera, confirm permissions and that no other app is using it."
                )
            if consecutive_failures >= 30:
                raise SystemExit("Stopping after repeated frame capture failures.")
            continue
        consecutive_failures = 0

        results = model(img, stream=True, verbose=False)

        total_score = 0

        # Coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])
                class_name = classNames[cls]
                print("Class name -->", class_name)

                # Add card value to total score
                total_score += card_values.get(class_name, 0)

                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                display_text = class_name if not SHOW_CONFIDENCE else f"{class_name} {confidence}"
                cv2.putText(img, display_text, org, font, fontScale, color, thickness)

        # Display total score on the screen
        score_text = f"Total Score: {total_score}"
        cv2.putText(img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window_title, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            SHOW_CONFIDENCE = not SHOW_CONFIDENCE
finally:
    cap.release()
    cv2.destroyAllWindows()
