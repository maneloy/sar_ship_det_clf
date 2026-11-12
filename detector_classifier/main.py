from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from interface import (
    detector_model,
    classifier_model,
    detect_and_classify
)

app = FastAPI(
    title="SAR Ship Detection API",
    description="Detects and classifies ships in SAR images."
)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    # Save uploaded image temporarily
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:

        detections = detect_and_classify(
            temp_path,
            detector_model,
            classifier_model,
            conf=0.6,
            padding=20
        )

        return {
            "num_detections": len(detections),
            "ships": detections
        }

    finally:
        os.remove(temp_path)