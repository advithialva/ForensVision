"""
Weapon Detection Model Handler
"""

import logging
import numpy as np
import torch
import gc
import time
from pathlib import Path
from typing import Dict, List, Any

# Import config
from config import Config

# YOLOv8 for weapon detection
try:
    from ultralytics import YOLO
except ImportError:
    logging.error("Ultralytics not found. Please install ultralytics")
    raise

logger = logging.getLogger(__name__)

class WeaponDetector:
    
    def __init__(self):
        logger.info("Loading weapon detection model...")
        
        # Load YOLO model for weapon detection
        model_path = Config.WEAPON_MODEL_PATH
        if not model_path.exists():
            logger.error(f"Weapon model not found at {model_path}")
            raise FileNotFoundError(f"Weapon model not found at {model_path}")
        
        self.model = YOLO(str(model_path))
        
        # Setup device acceleration
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Define weapon classes to detect
        self.weapon_classes = Config.WEAPON_DETECTION["weapon_classes"]
        
        # Multi-pass detection configurations from config
        self.detection_configs = Config.WEAPON_DETECTION["detection_configs"]
        
        logger.info(f"✅ Weapon detection model loaded (Device: {self.device.upper()})")
    
    def _estimate_processing_time(self, total_frames: int, fps: float) -> Dict[str, float]:
        # Based on empirical testing with different video lengths
        config_params = Config.WEAPON_DETECTION
        frame_skip = config_params["frame_skip"]
        max_frames = config_params["max_frames"]
        
        frames_to_process = min(total_frames // frame_skip, max_frames)
        
        # Estimated processing speeds (frames per second) from config
        fps_config = config_params["estimated_fps"]
        if self.device == 'mps':
            estimated_fps = fps_config["mps"]
        else:
            estimated_fps = fps_config["cpu"]
        
        estimated_time = frames_to_process / estimated_fps
        video_duration = total_frames / fps if fps > 0 else 0
        
        return {
            "estimated_processing_time": estimated_time,
            "video_duration": video_duration,
            "frames_to_process": frames_to_process,
            "processing_speed_fps": estimated_fps
        }
    
    async def detect_weapons(self, weapon_frames: Dict[str, List[np.ndarray]], confidence_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Main weapon detection function
        
        Args:
            weapon_frames: Dictionary containing pre-processed frames at different scales
                          {'raw': [...], '320': [...], '416': [...], '640': [...]}
            confidence_threshold: Minimum confidence for general detections (pistol uses 0.5)
            
        Returns:
            Dictionary containing detection results, timing, and confidence scores
        """
        # Get frame count from raw frames
        total_frames = len(weapon_frames.get('raw', []))
        logger.info(f"🔫 Weapon detection: {total_frames} frames")
        
        detected_weapons = {}
        
        # Video processing parameters from config
        config_params = Config.WEAPON_DETECTION
        max_frames = config_params["max_frames"]
        
        # Estimate processing time
        time_estimates = self._estimate_processing_time(total_frames, 30) # Assuming 30fps for estimation
        
        processed_frames = 0
        start_time = time.time()
        
        # Get pre-processed frames at different scales
        raw_frames = weapon_frames.get('raw', [])
        frames_320 = weapon_frames.get('320', [])
        frames_416 = weapon_frames.get('416', [])
        frames_640 = weapon_frames.get('640', [])
        
        try:
            for frame_idx in range(min(len(raw_frames), max_frames)):
                processed_frames += 1
                
                frame_detections = {}
                best_confidences = {}
                
                for i, config in enumerate(self.detection_configs):
                    # Use pre-processed frames 
                    if config["imgsz"] == 320:
                        current_frame = frames_320[frame_idx] if frame_idx < len(frames_320) else raw_frames[frame_idx]
                    elif config["imgsz"] == 640:
                        current_frame = frames_640[frame_idx] if frame_idx < len(frames_640) else raw_frames[frame_idx]
                    else:  # 416
                        current_frame = frames_416[frame_idx] if frame_idx < len(frames_416) else raw_frames[frame_idx]
                    
                    results = self.model.predict(source=current_frame, verbose=False, **config)
                    
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = self.model.names[cls_id]
                        
                        if name in self.weapon_classes:
                            # Track best confidence for each weapon across all passes
                            if name not in best_confidences or conf > best_confidences[name]:
                                best_confidences[name] = conf
                                frame_detections[name] = conf
                    
                    # Memory cleanup
                    del results
                
                # Apply weapon-specific thresholds and update global detections
                weapon_thresholds = Config.WEAPON_DETECTION["weapon_thresholds"]
                for weapon, conf in frame_detections.items():
                    if weapon == "pistol":
                        min_conf = 0.4 
                    elif weapon == "knife":
                        min_conf = 0.15  
                    elif weapon == "rifle":
                        min_conf = 0.2   
                    else:
                        min_conf = confidence_threshold
                    
                    if conf >= min_conf:
                        if weapon not in detected_weapons or conf > detected_weapons[weapon]:
                            detected_weapons[weapon] = conf
                
                # EARLY EXIT - but only for very high confidence detections
                if any(conf > 0.8 for conf in detected_weapons.values()):
                    logger.info(f"Early exit: High confidence weapon detected")
                    break
                
                # Only exit early for very confident detections to improve recall
                if len(detected_weapons) > 0 and max(detected_weapons.values()) > 0.85:
                    logger.info(f"Stopping: Very confident detection")
                    break
        
        finally:
            gc.collect() 
        
        # Calculate final timing statistics
        elapsed_time = time.time() - start_time
        avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
        
        # Prepare results
        weapons_found = len(detected_weapons) > 0
        
        # Create detailed confidence scores info
        confidence_info = {}
        weapon_thresholds = Config.WEAPON_DETECTION["weapon_thresholds"]
        for weapon, confidence in detected_weapons.items():
            # Get appropriate threshold for this weapon
            detection_threshold = weapon_thresholds.get(weapon, confidence_threshold)
            
            confidence_info[weapon] = {
                "confidence": round(confidence, 3),
                "confidence_level": self._get_confidence_level(confidence),
                "detection_threshold": detection_threshold
            }
        
        processing_stats = {
            "processing_time_seconds": round(elapsed_time, 2),
            "video_duration_seconds": round(time_estimates['video_duration'], 2),
            "total_frames": total_frames,
            "frames_processed": processed_frames,
            "frame_skip_interval": Config.WEAPON_DETECTION["frame_skip"],
            "average_processing_fps": round(avg_fps, 2),
            "device_used": self.device.upper(),
            "estimated_vs_actual_time": {
                "estimated": round(time_estimates['estimated_processing_time'], 2),
                "actual": round(elapsed_time, 2),
                "accuracy_percentage": round((time_estimates['estimated_processing_time'] / elapsed_time * 100), 1) if elapsed_time > 0 else 0
            }
        }
        
        result = {
            "weapons_detected": weapons_found,
            "detected_weapons": list(detected_weapons.keys()),
            "confidence_scores": confidence_info,
            "processing_stats": processing_stats,
            "raw_detections": detected_weapons,  # For internal use
            "analysis_summary": self._generate_analysis_summary(detected_weapons, processing_stats)
        }
        
        # Log final result
        if weapons_found:
            weapon_list = ", ".join([f"{w}({conf:.3f})" for w, conf in detected_weapons.items()])
            logger.info(f"🔫 Weapons: ⚠️ DETECTED ({weapon_list}) [{elapsed_time:.1f}s]")
        else:
            logger.info(f"🔫 Weapons: ✅ NONE DETECTED [{elapsed_time:.1f}s]")
        
        return result
    
    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Medium"
        elif confidence >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_analysis_summary(self, detected_weapons: Dict[str, float], processing_stats: Dict[str, Any]) -> Dict[str, Any]:
        if not detected_weapons:
            return {
                "status": "No weapons detected",
                "message": "The video has been analyzed and no weapons were found.",
                "processing_efficiency": f"Completed in {processing_stats['processing_time_seconds']}s using {processing_stats['device_used']}"
            }
        
        # Sort weapons by confidence
        sorted_weapons = sorted(detected_weapons.items(), key=lambda x: x[1], reverse=True)
        highest_conf_weapon = sorted_weapons[0]
        
        return {
            "status": "Weapons detected",
            "message": f"Detected {len(detected_weapons)} weapon type(s) in the video.",
            "highest_confidence_detection": {
                "weapon": highest_conf_weapon[0],
                "confidence": round(highest_conf_weapon[1], 3),
                "confidence_level": self._get_confidence_level(highest_conf_weapon[1])
            },
            "processing_efficiency": f"Completed in {processing_stats['processing_time_seconds']}s using {processing_stats['device_used']} ({processing_stats['average_processing_fps']:.1f} FPS)",
            "total_detections": len(detected_weapons)
        }