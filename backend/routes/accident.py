from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO

model_path = "model/yolo_accident.pt"  # Update with the correct path to your model
model = YOLO(model_path)

router = APIRouter()

@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        image_bytes = io.BytesIO(contents)

        # Open image using PIL
        img = Image.open(image_bytes).convert("RGB")

        # Convert PIL image to OpenCV format
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Perform inference using YOLOv8
        results = model.predict(img_cv, conf=0.3)

        # Draw bounding boxes on the image
        for result in results:
            img_cv = result.plot()  # Use YOLO's built-in visualization

        # Convert the processed image back to JPEG format
        _, encoded_img = cv2.imencode('.jpg', img_cv)
        img_bytes = io.BytesIO(encoded_img.tobytes())

        # Return the processed image as a response
        return StreamingResponse(img_bytes, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
