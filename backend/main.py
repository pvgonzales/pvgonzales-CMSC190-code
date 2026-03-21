import os
import io
import json
import base64
import asyncio
import tempfile
import logging
from typing import Optional
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from load_model import ModelManager, CLASS_NAMES, CLIP_LENGTH

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BYOL and DINOv3 Integration",
    description="Crime classification using BYOL and DINOv3 models",)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    global model_manager
    if model_manager is None:
        models_dir = os.environ.get("MODELS_DIR", "model")
        model_manager = ModelManager(models_dir=models_dir)
    return model_manager

# list the models and their metadata
@app.get("/api/models")
async def list_models():
    mm = get_model_manager()
    return {
        "models": mm.get_available_models(),
        "class_names": CLASS_NAMES,
        "clip_length": CLIP_LENGTH,
    }

#predict the category of crimes from the clip/frames
#request body: {"model": "byol" | "dino", "frames": [list of frames]}
@app.post("/api/predict")
async def predict_frames(data: dict):
    mm = get_model_manager()

    model_id = data.get("model", "byol")
    frames_b64 = data.get("frames", [])

    if len(frames_b64) != CLIP_LENGTH:
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected {CLIP_LENGTH} frames, got {len(frames_b64)}"},
        )
        
    pil_frames = []
    for b64_str in frames_b64:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pil_frames.append(img)

    try:
        result = mm.predict_clip(model_id, pil_frames)
        return result
    except Exception as e:
        logger.error(f"Prediction error with model '{model_id}': {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Model inference failed: {str(e)}"},
        )

#upload a video file and get predictions for every 16 frames
@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    model: str = Query(default="byol"),
    stride: int = Query(default=8, description="Frame stride between clips"),
):
    mm = get_model_manager()
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse(
                status_code=400,
                content={"error": "Could not open video file"},
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            all_frames.append(pil_frame)
        cap.release()

        if len(all_frames) < CLIP_LENGTH:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Video too short. Need at least {CLIP_LENGTH} frames"
                },
            )

        timeline = []
        clip_starts = range(0, len(all_frames) - CLIP_LENGTH + 1, stride)

        try:
            for start_idx in clip_starts:
                clip_frames = all_frames[start_idx : start_idx + CLIP_LENGTH]
                result = mm.predict_clip(model, clip_frames)

                mid_frame = start_idx + CLIP_LENGTH // 2
                timestamp = mid_frame / fps

                timeline.append({
                    "start_frame": start_idx,
                    "end_frame": start_idx + CLIP_LENGTH,
                    "timestamp": round(timestamp, 2),
                    "timestamp_str": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                    **result,
                })
        except Exception as e:
            logger.error(f"Video analysis failed with model '{model}': {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Model inference failed: {str(e)}"},
            )

        crime_predictions = [t for t in timeline if t['is_crime']]
        crime_ratio = len(crime_predictions) / len(timeline) if timeline else 0

        return {
            "video_info": {
                "filename": file.filename,
                "total_frames": len(all_frames),
                "fps": round(fps, 2),
                "duration": round(duration, 2),
                "duration_str": f"{int(duration // 60)}:{int(duration % 60):02d}",
            },
            "timeline": timeline,
            "summary": {
                "total_clips": len(timeline),
                "crime_clips": len(crime_predictions),
                "crime_ratio": round(crime_ratio, 3),
                "most_common_prediction": max(
                    set(t['prediction'] for t in timeline),
                    key=lambda c: sum(1 for t in timeline if t['prediction'] == c),
                ) if timeline else "N/A",
            },
        }

    finally:
        os.unlink(tmp_path)