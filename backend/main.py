from fastapi import FastAPI
from routes import stgcn, dcrnn, accident, helmet, vehicle  # Ensure these exist
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Initialize FastAPI App
app = FastAPI(title="AI Detection API", description="Detects various objects and incidents", version="1.0")

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Include API routers
app.include_router(stgcn.router, prefix="/stgcn", tags=["STGCN"])
app.include_router(dcrnn.router, prefix="/dcrnn", tags=["DCRNN"])
app.include_router(accident.router, prefix="/accident", tags=["Accident Detection"])
app.include_router(helmet.router, prefix="/helmet", tags=["Helmet Detection"])
app.include_router(vehicle.router, prefix="/vehicle", tags=["Vehicle Detection"])

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "API is running"}

# Run FastAPI server
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
