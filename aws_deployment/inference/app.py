import os
import sys
import io
import time
import boto3
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Ensure repository imports work
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'classification'))

try:
    from classification.config import get_config
    from classification.models import build_model
    from classification.data.build import build_transform
except ImportError as e:
    # Fallback to local import structure if files are in execution context
    from config import get_config
    from models import build_model
    from data.build import build_transform

app = FastAPI(
    title="TIH Crop Classification API",
    description="Inference Service for DAMamba Vision State Space Models on Crop Health.",
    version="1.0.0"
)

# Enable CORS for web frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded class mapping based on training metrics
DEFAULT_CLASSES = [
    "Alternaria_Blight",
    "Aphid",
    "Healthy",
    "Pea_Leaf_Miner",
    "Powdery_Mildew",
    "Unknown",
    "White_Rust_Leaf"
]

# Global variables to hold model, transform and device
model = None
transform = None
device = None
class_names = DEFAULT_CLASSES

# Define request classes if needed or response models
class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    device: str
    classes: List[str]

def download_model_from_s3(bucket: str, key: str, local_path: str):
    """Downloads model checkpoint from Amazon S3."""
    print(f"Downloading checkpoint from S3: s3://{bucket}/{key} to {local_path}...")
    s3 = boto3.client('s3')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)
    print("Download complete!")

@app.on_event("startup")
def load_ml_model():
    global model, transform, device, class_names
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")
    
    # Get config path and checkpoint path from environment variables
    config_path = os.environ.get(
        "MODEL_CONFIG", 
        os.path.join(root_dir, "classification", "configs", "DAMamba", "damamba_tiny.yaml")
    )
    checkpoint_path = os.environ.get(
        "MODEL_CHECKPOINT", 
        os.path.join(root_dir, "output", "tiny", "finetune", "best_ckpt.pth")
    )
    
    # Download weights from S3 if configured and not present locally
    s3_bucket = os.environ.get("MODEL_S3_BUCKET")
    s3_key = os.environ.get("MODEL_S3_KEY")
    if s3_bucket and s3_key and not os.path.exists(checkpoint_path):
        try:
            download_model_from_s3(s3_bucket, s3_key, checkpoint_path)
        except Exception as e:
            print(f"Error downloading weights from S3: {e}. Expecting local file...")
            
    # Load custom class names if provided via env
    env_classes = os.environ.get("CLASS_NAMES")
    if env_classes:
        class_names = [c.strip() for c in env_classes.split(",")]
        print(f"Loaded class mapping from environment: {class_names}")
    else:
        print(f"Using default class mapping: {class_names}")

    # Build config node
    class TempArgs:
        def __init__(self, cfg):
            self.cfg = cfg
            self.opts = None
            self.batch_size = 1
            self.data_path = None
            self.zip = False
            self.cache_mode = None
            self.pretrained = None
            self.resume = checkpoint_path
            self.accumulation_steps = None
            self.use_checkpoint = False
            self.disable_amp = False
            self.output = None
            self.tag = None
            self.oversample = None
            self.eval = True
            self.throughput = False
            self.traincost = False
            self.enable_persistance = False
            self.enable_amp = False
            self.fused_layernorm = False
            self.optim = None
            self.ddp = 'torch'

    temp_args = TempArgs(config_path)
    config = get_config(temp_args)

    # Overwrite number of classes based on loaded configuration or checkpoint
    config.defrost()
    config.MODEL.NUM_CLASSES = len(class_names)
    config.freeze()

    # Build model structure
    print(f"Building model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Determine weight key
        if 'model' in checkpoint:
            weights = checkpoint['model']
        elif 'model_ema' in checkpoint:
            weights = checkpoint['model_ema']
        else:
            weights = checkpoint
            
        # Clean 'module.' DDP prefix
        cleaned_weights = {}
        for k, v in weights.items():
            key_name = k[7:] if k.startswith('module.') else k
            cleaned_weights[key_name] = v
            
        msg = model.load_state_dict(cleaned_weights, strict=True)
        print(f"Weights loaded: {msg}")
    else:
        print(f"Warning: Checkpoint file '{checkpoint_path}' not found. Starting with random weights.")

    model = model.to(device)
    model.eval()

    # Build image transforms
    transform = build_transform(is_train=False, config=config)
    print("Inference engine initialized successfully!")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify service and model readiness."""
    if model is None or transform is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return HealthResponse(
        status="healthy",
        device=str(device),
        classes=class_names
    )

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """Receives an image and returns the predicted crop class and confidence score."""
    if model is None or transform is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")

    start_time = time.time()
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Transform and move to device
        with torch.no_grad():
            tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            outputs = model(tensor)
            probabilities = nn.functional.softmax(outputs, dim=-1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=-1)
            
            class_name = class_names[predicted_idx.item()]
            
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictionResult(
            class_name=class_name,
            confidence=float(confidence.item()),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
