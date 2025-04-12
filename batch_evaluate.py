import os
import requests
import csv

# Configuration
IMAGE_FOLDER = r"F:\annotatedImage\inputpics"
SAVE_FOLDER = r"F:\annotatedImage\outputpics"
API_URL = "http://127.0.0.1:8001/estimate-height"

# Output CSV
report_path = os.path.join(SAVE_FOLDER, "accuracy_report.csv")

# Prepare CSV headers
with open(report_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "actual_height", "estimated_height", "error_cm"])

# Process each image
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):

        image_path = os.path.join(IMAGE_FOLDER, filename)
        name_base = os.path.splitext(filename)[0]
        actual_height_path = os.path.join(IMAGE_FOLDER, name_base + ".txt")

        actual_height = None
        if os.path.exists(actual_height_path):
            with open(actual_height_path, "r") as f:
                try:
                    actual_height = float(f.read().strip())
                except ValueError:
                    print(f"❌ Could not parse height for {filename}")
                    continue

        # Send image to FastAPI
        with open(image_path, "rb") as img_file:
            response = requests.post(API_URL, files={"image": img_file})

        if response.status_code == 200:
            data = response.json()
            est_height = data.get("estimated_height_cm", None)

            if est_height and actual_height:
                error = abs(est_height - actual_height)
                print(f"✅ {filename}: actual={actual_height}, estimated={est_height:.2f}, error={error:.2f}cm")

                with open(report_path, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([filename, actual_height, round(est_height, 2), round(error, 2)])
            else:
                print(f"⚠️ {filename}: missing values")
        else:
            print(f"❌ Failed: {filename} → Status {response.status_code}")

print(f"\n📊 Report saved to: {report_path}")
