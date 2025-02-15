from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os



model_path = "model/helmet.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")

try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None
router = APIRouter()
@router.post("/helmet/detect/")
async def detect_helmet(image: UploadFile = File(...)):
    try:
        if model is None:
            return JSONResponse({"error": "YOLO model not loaded"}, status_code=500)

        image_bytes = await image.read()
        print(f"✅ Received image: {len(image_bytes)} bytes")

        try:
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return JSONResponse({"error": f"Invalid image file: {e}"}, status_code=400)

        image_np = np.array(image_pil)

        print(f"✅ Image shape: {image_np.shape}")  # Check if the image is valid
        results = model(image_np)
        print(f"✅ Detection results: {results}")

        helmet_count = 0
        no_helmet_count = 0
        detections = []

        for result in results:
            for box in result.boxes:
                try:
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    if label == "helmet":
                        helmet_count += 1
                    else:
                        no_helmet_count += 1

                    detections.append({
                        "x1": int(box.xyxy[0][0]),
                        "y1": int(box.xyxy[0][1]),
                        "x2": int(box.xyxy[0][2]),
                        "y2": int(box.xyxy[0][3]),
                        "label": label
                    })
                except Exception as e:
                    print(f"⚠️ Error processing detection box: {e}")
        
        return JSONResponse({
            "helmet_count": helmet_count,
            "no_helmet_count": no_helmet_count,
            "detections": detections
        })

    except Exception as e:
        print(f"🔥 Error in detection: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
