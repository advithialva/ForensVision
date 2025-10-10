from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from datetime import datetime
import cv2
import numpy as np
import tempfile
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncContextManager
import asyncio
from contextlib import asynccontextmanager
import json

# Import our model handlers
from models.violence_detector import ViolenceDetector
from models.weapon_detector import WeaponDetector
from utils.video_processor import VideoProcessor
from utils.response_formatter import ResponseFormatter


class NumpyJSONResponse(JSONResponse):
    """
    Custom JSONResponse to handle NumPy data types during serialization.
    """
    def render(self, content: Any) -> bytes:
        return json.dumps(content, cls=NumpyEncoder).encode("utf-8")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def save_upload_file_tmp(upload_file: UploadFile):
    """
    Asynchronous context manager to save an UploadFile to a temporary file.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload_file.filename).suffix) as tmp:
            content = await upload_file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        yield tmp_path
    finally:
        if 'tmp_path' in locals() and tmp_path.exists():
            os.remove(tmp_path)

# Global model instances
violence_detector = None
weapon_detector = None
video_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup and cleanup on shutdown"""
    global violence_detector, weapon_detector, video_processor
    
    try:
        logger.info("🚀 Initializing ForensVision models...")
        
        # Initialize detectors
        violence_detector = ViolenceDetector()
        weapon_detector = WeaponDetector()
        video_processor = VideoProcessor()
        
        logger.info("✅ All models loaded successfully!")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        raise
    finally:
        logger.info("🔄 Cleaning up resources...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="ForensVision API",
    description="Unified API for violence and weapon detection in videos",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ForensVision API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global violence_detector, weapon_detector
    
    return {
        "status": "healthy",
        "models": {
            "violence_detector": violence_detector is not None,
            "weapon_detector": weapon_detector is not None,
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    include_violence: bool = True,
    include_weapons: bool = True,
    confidence_threshold: float = 0.5
):
    """Analyze video for violence and weapon detection"""
    global violence_detector, weapon_detector, video_processor
    
    logger.info(f"🎬 Analysis Started: {file.filename}")
    
    async with save_upload_file_tmp(file) as temp_video_path:
        # Validate video file
        is_valid, error_msg = video_processor.validate_video_file(str(temp_video_path))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        video_info = video_processor.get_video_info(str(temp_video_path))
        logger.info(f"📊 Video: {video_info['duration_seconds']}s, {video_info['resolution']['width']}x{video_info['resolution']['height']}")
        
        tasks = []
        if include_violence:
            raw_frames, lstm_frames, multi_scale_frames = video_processor.extract_frames_for_violence(str(temp_video_path))
            tasks.append(violence_detector.detect_violence(raw_frames, lstm_frames, multi_scale_frames))
        
        if include_weapons:
            weapon_frames = video_processor.extract_frames_for_weapons(str(temp_video_path))
            tasks.append(weapon_detector.detect_weapons(weapon_frames, confidence_threshold))
        
        # DETECTION PHASE
        logger.info(f"🔍 Running {len(tasks)} detection model(s)...")
        results = await asyncio.gather(*tasks)
        
        # RESULTS PROCESSING
        violence_result = None
        weapon_result = None
        
        result_index = 0
        if include_violence:
            violence_result = results[result_index]
            result_index += 1
        if include_weapons:
            weapon_result = results[result_index]
            
        summary_parts = []
        if violence_result:
            is_violence = violence_result.get('is_violence', False)
            confidence = violence_result.get('confidence', 0.0)
            summary_parts.append(f"Violence: {'⚠️ DETECTED' if is_violence else '✅ SAFE'} ({confidence:.1%})")
        
        if weapon_result:
            weapons_detected = weapon_result.get('weapons_detected', False)
            detected_weapons = weapon_result.get('detected_weapons', [])
            if weapons_detected and detected_weapons:
                weapons_str = ', '.join(detected_weapons)
                summary_parts.append(f"Weapons: ⚠️ DETECTED ({weapons_str})")
            else:
                summary_parts.append("Weapons: ✅ NONE")
        
        if summary_parts:
            logger.info(f"🎯 Results: {' | '.join(summary_parts)}")
        
        detection_results = ResponseFormatter.format_detection_results(
            violence_result=violence_result,
            weapon_result=weapon_result
        )
        
        response = {
            "filename": file.filename,
            "video_info": video_info,
            "analysis": {
                "violence_detection": detection_results["violence_analysis"],
                "weapon_detection": detection_results["weapon_analysis"]
            },
            "summary": detection_results["summary"],
            "analysis_params": {
                "include_violence": include_violence,
                "include_weapons": include_weapons,
                "confidence_threshold": confidence_threshold,
                "processing_method": "unified_preprocessing"
            }
        }
        
        logger.info("✅ Analysis Complete")
        return NumpyJSONResponse(content=response)


@app.post("/analyze/violence")
async def analyze_violence_only(
    file: UploadFile = File(...)
):
    """
    Analyze video for violence detection only
    
    Args:
        file: Video file to analyze
    
    Returns:
        Violence detection results
    """
    global violence_detector, video_processor
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    async with save_upload_file_tmp(file) as tmp_file_path:
        logger.info(f"🧠 Analyzing violence in: {file.filename}")
        
        # Extract frames optimized for violence detection
        raw_frames, lstm_frames, multi_scale_frames = video_processor.extract_frames_for_violence(str(tmp_file_path))
        if len(raw_frames) == 0:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")

        # Get video info
        video_info = video_processor.get_video_info(str(tmp_file_path))
        
        # Run violence detection on frames
        violence_result = await violence_detector.detect_violence(raw_frames, lstm_frames, multi_scale_frames)
        
        result = {
            "filename": file.filename,
            "video_info": video_info,
            "violence_detection": violence_result
        }
        
        # Return response using custom JSON encoder
        return NumpyJSONResponse(content=result)

@app.post("/analyze/weapons")
async def analyze_weapons_only(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    """
    Analyze video for weapon detection only
    
    Args:
        file: Video file to analyze
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        Weapon detection results
    """
    global weapon_detector, video_processor
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    async with save_upload_file_tmp(file) as tmp_file_path:
        logger.info(f"🔫 Analyzing weapons in: {file.filename}")
        
        # Extract frames optimized for weapon detection
        weapon_frames = video_processor.extract_frames_for_weapons(str(tmp_file_path))
        if len(weapon_frames.get('raw', [])) == 0:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")

        # Get video info
        video_info = video_processor.get_video_info(str(tmp_file_path))
        
        # Run weapon detection on frames
        weapon_result = await weapon_detector.detect_weapons(weapon_frames, confidence_threshold)
        
        result = {
            "filename": file.filename,
            "video_info": video_info,
            "weapon_detection": weapon_result
        }
        
        # Return response using custom JSON encoder
        return NumpyJSONResponse(content=result)

@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    global violence_detector, weapon_detector
    
    return {
        "models": {
            "violence_detection": {
                "loaded": violence_detector is not None,
                "model_type": "MoBiLSTM + YOLOv8 + ResNet-50",
                "capabilities": ["behavioral_analysis", "person_tracking", "visual_features"]
            },
            "weapon_detection": {
                "loaded": weapon_detector is not None,
                "model_type": "YOLOv8",
                "capabilities": ["real_time_detection", "multiple_weapon_types"]
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)