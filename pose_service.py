from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np

app = FastAPI()
mp_pose = mp.solutions.pose

@app.post("/estimate-height")
async def estimate_height(image: UploadFile = File(...)):
    # Read image and convert to OpenCV format
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run MediaPipe Pose detection
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return JSONResponse(content={"error": "No person detected"}, status_code=400)

        landmarks = results.pose_landmarks.landmark

        # Get head (nose) and lowest heel
        head_y = landmarks[mp_pose.PoseLandmark.NOSE].y * img.shape[0]
        heel_y = min(
            landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y
        ) * img.shape[0]

        pixel_height = abs(heel_y - head_y)

    # Detect the reference marker (e.g., A4 paper)
    marker_height_px = detect_marker_height(img)
    if not marker_height_px:
        return JSONResponse(
            content={"error": "Reference marker not found."},
            status_code=400
        )

    # Convert pixel height to cm using A4 paper height (29.7 cm)
    cm_per_pixel = 29.7 / marker_height_px
    person_height_cm = pixel_height * cm_per_pixel

    return {
        "pixel_height": pixel_height,
        "marker_height_px": marker_height_px,
        "cm_per_pixel": cm_per_pixel,
        "estimated_height_cm": round(person_height_cm, 2)
    }


def detect_marker_height(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marker_height_pixels = None

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:  # likely a rectangle
            (x, y, w, h) = cv2.boundingRect(approx)
            marker_height_pixels = h
            break  # take the first one found (optional: choose largest)

    return marker_height_pixels
