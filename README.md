# Height Estimator - Python FastAPI Service

This is a lightweight FastAPI microservice that accepts an image upload and returns a mocked height estimation in centimeters with a confidence score.

## 📦 Features

- Accepts image uploads via `multipart/form-data`
- Parses and prints image metadata (filename, size)
- Returns a mocked JSON response with:
  - `status`
  - `estimated_height_cm`
  - `confidence`

## 🚀 How to Run

### 1. Install dependencies
We recommend using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
