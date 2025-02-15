import io
import cv2
import base64
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# FastAPI Router
router = APIRouter()

# Load YOLO model
model = YOLO("model/vehical.pt")  # Ensure your model file is in the correct directory

# Define vehicle classes
CLASSES = ['auto', 'bus', 'car', 'lcv', 'motorcycle', 'multiaxle', 'tractor', 'truck']

def process_image(image_bytes):
    """Processes a single image and returns vehicle count and the processed image."""
    np_img = np.array(Image.open(io.BytesIO(image_bytes)))

    # Run YOLO model
    detections = model(np_img)

    # Initialize count dictionary
    vehicle_count = {cls: 0 for cls in CLASSES}

    # Draw bounding boxes on the image
    for result in detections:
        for box in result.boxes:
            cls_id = int(box.cls)
            vehicle_name = CLASSES[cls_id]
            vehicle_count[vehicle_name] += 1

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{vehicle_name} ({box.conf[0]:.2f})"

            # Draw rectangle & label
            cv2.rectangle(np_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(np_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert image to base64
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
    processed_image_base64 = base64.b64encode(buffer).decode("utf-8")

    # Filter out zero-count classes
    detected_vehicles = {k: v for k, v in vehicle_count.items() if v > 0}

    return {
        "total_vehicles": sum(detected_vehicles.values()),
        "vehicles_detected": detected_vehicles,
        "processed_image": processed_image_base64  # Return image as base64
    }

@router.post("/detect/")
async def detect_vehicles(images: list[UploadFile] = File(...)):
    if len(images) != 4:
        return JSONResponse(status_code=400, content={"error": "Exactly 4 images are required."})

    results = {}
    for i, image in enumerate(images):
        image_bytes = await image.read()
        results[f"image_{i+1}"] = process_image(image_bytes)

    return JSONResponse(content=results)
