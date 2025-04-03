from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    print(f"✅ Received file: {file.filename} ({len(contents)} bytes)")

    # Mocked response - this is where you'd plug in AI later
    return JSONResponse(content={
        "status": "success",
        "estimated_height_cm": 173.6,
        "confidence": 0.93
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
