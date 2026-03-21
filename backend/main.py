import os
import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
