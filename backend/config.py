import os
from pathlib import Path
from typing import Dict, Any

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent
    PROJECT_ROOT = BASE_DIR.parent  
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Model paths
    VIOLENCE_MODEL_PATH = MODELS_DIR / "violence_detection" / "MoBiLSTM_violence_detection_model.h5"
    WEAPON_MODEL_PATH = MODELS_DIR / "weapon_detection" / "weapon_detection.pt"
    YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # File upload settings
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 100))
    ALLOWED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.mpeg', '.mpg', '.webm'}
    
    # Processing settings
    DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", 0.5))
    MAX_PROCESSING_TIME_SECONDS = int(os.getenv("MAX_PROCESSING_TIME_SECONDS", 300))
    
    # Violence detection settings
    VIOLENCE_DETECTION = {
        "image_height": 64,
        "image_width": 64,
        "sequence_length": 16,
        "classes": ["NonViolence", "Violence"],
        "ensemble_weights": {
            'lstm': 0.55,
            'temporal': 0.15,
            'spatial': 0.12,
            'interaction': 0.10,
            'visual': 0.08
        },
        "base_threshold": 0.32,
        "threshold_adjustments": {
            "strong_agreement_factor": 0.75,
            "good_agreement_factor": 0.85,
            "high_confidence_factor": 0.8,
            "high_uncertainty_factor": 1.15,
            "confidence_boost_factor": 1.2,
            "confidence_boost_threshold_factor": 0.9
        },
        "feature_normalization": {
            "motion_intensity_divisor": 50.0,
            "motion_variability_divisor": 20.0,
            "motion_peaks_divisor": 5.0,
            "spatial_distribution_divisor": 2.0,
            "texture_intensity_divisor": 1000.0,
            "edge_density_multiplier": 10.0,
            "saturation_var_divisor": 10000,
            "value_var_divisor": 10000
        },
        "confidence_thresholds": {
            "high_confidence_min": 0.5,
            "low_uncertainty_max": 0.1,
            "uncertainty_penalty_threshold": 0.2,
            "low_confidence_max": 0.2,
            "lstm_boost_threshold": 0.7,
            "temporal_boost_threshold": 0.5,
            "borderline_min": 0.25,
            "borderline_max": 0.45
        },
        "weight_adjustments": {
            "lstm_max_boost": 0.75,
            "lstm_boost_amount": 0.15,
            "lstm_min_reduction": 0.35,
            "lstm_reduction_amount": 0.15,
            "remaining_weight_base": 0.45
        }
    }
    
    # Weapon detection settings
    WEAPON_DETECTION = {
        "confidence_threshold": 0.5,
        "iou_threshold": 0.5,
        "image_size": 320,
        "weapon_classes": {"pistol", "knife", "rifle"},
        "frame_skip": 8,
        "max_frames": 75,
        "early_exit_confidence": 0.85,
        "weapon_thresholds": {
            "pistol": 0.4,
            "knife": 0.15,
            "rifle": 0.2
        },
        "detection_configs": [
            {"conf": 0.05, "iou": 0.3, "imgsz": 416},
            {"conf": 0.1, "iou": 0.4, "imgsz": 640},
            {"conf": 0.08, "iou": 0.25, "imgsz": 320},
        ],
        "estimated_fps": {
            "mps": 18.0,
            "cpu": 12.0
        }
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(LOGS_DIR / "forensvision.log"),
                "mode": "a",
            },
        },
        "loggers": {
            "": {
                "level": "INFO",
                "handlers": ["console", "file"],
            },
        },
    }
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        (cls.MODELS_DIR / "violence_detection").mkdir(exist_ok=True)
        (cls.MODELS_DIR / "weapon_detection").mkdir(exist_ok=True)