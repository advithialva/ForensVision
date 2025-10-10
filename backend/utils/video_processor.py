import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

class VideoProcessor:   
    def __init__(self):
        self.supported_formats = {
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', 
            '.mpeg', '.mpg', '.webm', '.flv', '.m4v'
        }
        
        logger.info("📹 Video processor initialized")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(video_path)
            try:
                if not cap.isOpened():
                    raise ValueError(f"Could not open video file: {video_path}")
                
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0
                
                # Get file info
                file_path = Path(video_path)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                
                video_info = {
                    "filename": file_path.name,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "duration_seconds": round(duration, 2),
                    "total_frames": total_frames,
                    "fps": round(fps, 2),
                    "resolution": {
                        "width": width,
                        "height": height,
                        "total_pixels": width * height
                    },
                    "format": file_path.suffix.lower(),
                    "is_supported": file_path.suffix.lower() in self.supported_formats
                }
                
                logger.info(f"📊 Video info extracted: {file_path.name} ({duration:.1f}s, {width}x{height})")
                return video_info
            finally:
                cap.release()
                
        except Exception as e:
            logger.error(f"❌ Error extracting video info: {e}")
            return {
                "filename": Path(video_path).name,
                "error": str(e),
                "is_supported": False
            }
    
    def validate_video_file(self, video_path: str) -> Tuple[bool, str]:
        try:
            file_path = Path(video_path)
            
            # Check if file exists
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                return False, f"Unsupported format: {file_path.suffix}"
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file"
            
            # Check if video has frames
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return False, "Video file appears to be corrupted or empty"
            
            return True, "Video file is valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _prepare_frames_for_lstm(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        from config import Config
        target_size = (Config.VIOLENCE_DETECTION["image_height"], Config.VIOLENCE_DETECTION["image_width"])
        lstm_frames = []
        for frame in frames:
            # LSTM-specific preprocessing: resize + normalize
            resized_frame = cv2.resize(frame, target_size)
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            lstm_frames.append(normalized_frame)
        return lstm_frames
    
    def _prepare_frames_for_yolo(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        # YOLO works best with raw frames - just copy to avoid reference issues
        return [frame.copy() for frame in frames]
    
    def extract_frames_for_weapons(self, video_path: str, frame_skip: int = 8, max_frames: int = 75) -> Dict[str, List[np.ndarray]]:
        """
        Extract frames optimized for weapon detection with multi-scale preparation
        
        Args:
            video_path: Path to the video file
            frame_skip: Skip every N frames for processing
            max_frames: Maximum frames to process
            
        Returns:
            Dictionary with different sized frames for weapon detection:
            - 'raw': Original frames
            - '320': Frames resized to 320x320
            - '416': Frames resized to 416x416  
            - '640': Frames resized to 640x640
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            raw_frames = []
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or processed_frames >= max_frames:
                    break
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                
                processed_frames += 1
                raw_frames.append(frame)
            
            cap.release()
            logger.info(f"📸 Weapon frames: {len(raw_frames)} frames extracted")
            
            # Prepare multi-scale frames for weapon detection
            weapon_frames = {
                'raw': self._prepare_frames_for_yolo(raw_frames),
                '320': [cv2.resize(frame, (320, 320)) for frame in raw_frames],
                '416': [cv2.resize(frame, (416, 416)) for frame in raw_frames], 
                '640': [cv2.resize(frame, (640, 640)) for frame in raw_frames]
            }
            
            return weapon_frames
            
        except Exception as e:
            logger.error(f"❌ Error extracting frames for weapons: {e}")
            raise
    
    def extract_frames_for_violence(self, video_path: str, sequence_length: int = 16) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[float, List[np.ndarray]]]:
        """
        Extract frames optimized for violence detection with multi-scale preparation
        
        Args:
            video_path: Path to the video file
            sequence_length: Number of frames for LSTM sequence
            
        Returns:
            Tuple of (raw_frames, lstm_frames, multi_scale_frames) for violence detection
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip = max(int(total_frames / sequence_length), 1)
            
            raw_frames = []
            
            for frame_num in range(sequence_length):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num * skip)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                raw_frames.append(frame)
            
            cap.release()
            logger.info(f"📸 Violence frames: {len(raw_frames)} frames extracted")
            
            # Use centralized preprocessing
            lstm_frames = self._prepare_frames_for_lstm(raw_frames)
            multi_scale_frames = self._prepare_multi_scale_frames(raw_frames)
            
            return raw_frames, lstm_frames, multi_scale_frames
            
        except Exception as e:
            logger.error(f"❌ Error extracting frames for violence: {e}")
            raise
    
    def _prepare_multi_scale_frames(self, frames: List[np.ndarray], scales: List[float] = [1.0, 0.75, 0.5]) -> Dict[float, List[np.ndarray]]:
        multi_scale_frames = {scale: [] for scale in scales}
        
        for frame in frames:
            for scale in scales:
                if scale != 1.0:
                    scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                else:
                    scaled_frame = frame.copy()  # Keep original
                multi_scale_frames[scale].append(scaled_frame)
        
        return multi_scale_frames