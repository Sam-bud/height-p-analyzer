from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import mediapipe as mp
import numpy as np
import base64
import datetime
import os

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

        # Detect the reference marker (e.g., A4 paper)
    marker_height_px = detect_marker_height(img)

        # 1. Extract landmarks related to upper face
    head_candidates = [
        landmarks[mp_pose.PoseLandmark.NOSE].y,
        landmarks[mp_pose.PoseLandmark.LEFT_EYE].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y,
        landmarks[mp_pose.PoseLandmark.LEFT_EAR].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y
    ]
    # 2. Find top-most facial point in Y-pixels
    raw_head_y = min(head_candidates) * img.shape[0]

    # 3. Compute cm-per-pixel scale using A4 height
    cm_per_pixel = 29.7 / marker_height_px  # A4 marker assumed

    # 4. Estimate real-world eye-to-hair distance (12 cm) → pixel offset
    eye_to_hair_cm = 12.0
    offset_px = eye_to_hair_cm / cm_per_pixel

    # 5. Adjust head_y upward to crown
    head_y = raw_head_y - offset_px

    heel_y = max(
        landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y
    ) * img.shape[0] - 10  # ← reduce overreach from shoes/floor

    pixel_height = abs(heel_y - head_y)


    if not marker_height_px:
        return JSONResponse(
            content={"error": "Reference marker not found."},
            status_code=400
        )

    # Convert pixel height to cm using A4 paper height (29.7 cm)
    cm_per_pixel = 29.7 / marker_height_px
    person_height_cm = pixel_height * cm_per_pixel

    # 🔵 Draw pose landmarks
    for landmark in landmarks:
     x = int(landmark.x * img.shape[1])
     y = int(landmark.y * img.shape[0])
     cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

     # 🔹 Draw head & heel lines
    cv2.line(img, (0, int(head_y)), (img.shape[1], int(head_y)), (255, 0, 0), 2)
    cv2.line(img, (0, int(heel_y)), (img.shape[1], int(heel_y)), (0, 0, 255), 2)

# 🔵 Draw height info
    cv2.putText(img, f"Height: {round(person_height_cm, 2)} cm", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# 🔹 Save to F:\annotatedImage
    filename = datetime.datetime.now().strftime("height_%Y%m%d_%H%M%S.jpg")
    save_path = os.path.join("F:\\annotatedImage", filename)
    cv2.imwrite(save_path, img)

    print(f"✅ Annotated image saved to: {save_path}")

# 🔵 Encode image to JPEG for response
    _, buffer = cv2.imencode('.jpg', img)
    annotated_bytes = BytesIO(buffer.tobytes())

    base64_image = base64.b64encode(buffer).decode("utf-8")

    print("📏 --- HEIGHT DEBUG ---")
    print(f"Head Y: {head_y}")
    print(f"Heel Y: {heel_y}")
    print(f"Pixel height (heel - head): {pixel_height}")
    print(f"Marker height in pixels: {marker_height_px}")
    print(f"cm_per_pixel = 29.7 / {marker_height_px} = {cm_per_pixel}")
    print(f"Estimated height (cm): {person_height_cm}")
    print("📏 --------------------")


    return {
        "estimated_height_cm": round(person_height_cm, 2)
         }



def detect_marker_height(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    a4_ratio = 29.7 / 21.0  # A4 height / width ≈ 1.41
    marker_height_pixels = None
    best_match_score = 0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ratio = h / w if w != 0 else 0
            area = w * h

            # Give high score to large areas with aspect ratio close to A4
            match_score = area / (1 + abs(ratio - a4_ratio))

            if match_score > best_match_score:
                best_match_score = match_score
                marker_height_pixels = h

    return marker_height_pixels

