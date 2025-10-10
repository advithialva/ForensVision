"""
Violence Detection Model Handler: Integrates MoBiLSTM, YOLOv8, and ResNet-50
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any
from sklearn.cluster import DBSCAN
from config import Config

# TensorFlow and deep learning imports
try:
    from keras.models import load_model
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    logging.error("TensorFlow not found. Please install tensorflow")
    raise

# YOLOv8 for person detection
try:
    from ultralytics import YOLO
except ImportError:
    logging.error("Ultralytics not found. Please install ultralytics")
    raise

# Hugging Face transformers for visual features
try:
    from transformers import AutoFeatureExtractor, AutoModel
    import torch
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("Transformers not available, visual features disabled")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class PersonTracker:
    """
    Person tracking system with movement history for behavioral analysis.
    """
    
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.movement_history = {}
        
    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.movement_history[self.next_id] = [centroid]
        self.next_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.movement_history:
            del self.movement_history[object_id]
            
    def update(self, rects):
        if len(rects) == 0:
            # Handle case when no persons detected
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        # Calculate centroids from bounding boxes
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)
            
        if len(self.objects) == 0:
            # Initialize tracking for first detection
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match existing persons with new detections
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            if len(object_centroids) > 0 and len(input_centroids) > 0:
                # Calculate distance matrix for matching
                D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                
                used_row_idxs = set()
                used_col_idxs = set()
                
                # Perform matching with distance threshold
                for (row, col) in zip(rows, cols):
                    if row in used_row_idxs or col in used_col_idxs:
                        continue
                        
                    if row >= len(object_ids) or D[row, col] > 100:  # Distance threshold
                        continue
                        
                    # Update matched person
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    
                    # Update movement history for behavioral analysis
                    if object_id in self.movement_history:
                        self.movement_history[object_id].append(input_centroids[col])
                        # Maintain sliding window of positions
                        if len(self.movement_history[object_id]) > 30:
                            self.movement_history[object_id] = self.movement_history[object_id][-30:]
                            
                    used_row_idxs.add(row)
                    used_col_idxs.add(col)
                    
                # Handle unmatched detections
                unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
                unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
                
                # Update disappeared persons or register new ones
                if D.shape[0] >= D.shape[1]:
                    for row in unused_row_idxs:
                        if row < len(object_ids):
                            object_id = object_ids[row]
                            self.disappeared[object_id] += 1
                            if self.disappeared[object_id] > self.max_disappeared:
                                self.deregister(object_id)
                else:
                    for col in unused_col_idxs:
                        if col < len(input_centroids):
                            self.register(input_centroids[col])
                    
        return self.objects

class ViolenceDetector:
    
    def __init__(self):
        self.sequence_length = Config.VIOLENCE_DETECTION["sequence_length"]
        self.classes = Config.VIOLENCE_DETECTION["classes"]
        
        logger.info("🧠 Loading violence detection models...")
        
        # Load MoBiLSTM model
        model_path = Config.VIOLENCE_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"MoBiLSTM model not found at {model_path}")
        
        self.mobilstm_model = load_model(str(model_path))
        logger.info("✅ MoBiLSTM model loaded")
        
        # Load YOLO for person detection
        yolo_path = Config.YOLO_MODEL_PATH
        if not yolo_path.exists():
            # Try alternative path
            yolo_path = Path(__file__).parent.parent / "notebooks" / "violence_detection" / "yolov8n.pt"
        
        self.yolo_model = YOLO(str(yolo_path) if yolo_path.exists() else 'yolov8n.pt')
        logger.info("✅ YOLO model loaded")
        
        # Initialize Hugging Face models for visual features
        self.hf_model_available = False
        if TRANSFORMERS_AVAILABLE:
            try:
                self.hf_processor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
                self.hf_model = AutoModel.from_pretrained("microsoft/resnet-50")
                self.hf_model.eval()
                self.hf_model_available = True
                logger.info("✅ ResNet-50 visual feature extractor loaded")
            except Exception as e:
                logger.warning(f"Visual feature extractor not available: {e}")
        
        logger.info("🚀 Violence detection system ready!")
    
    def extract_visual_features(self, frames):
        if not self.hf_model_available or len(frames) == 0:
            return 0.0
        
        try:
            # Convert frames to PIL Images and process
            feature_scores = []
            
            # Sample 4 key frames for efficiency
            frame_indices = [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]
            
            for idx in frame_indices:
                if idx < len(frames):
                    # Convert frame to PIL Image
                    frame = frames[idx]
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    pil_image = Image.fromarray(frame_uint8)
                    
                    # Process with Hugging Face model
                    inputs = self.hf_processor(images=pil_image, return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = self.hf_model(**inputs)
                        # Get pooled features
                        features = outputs.last_hidden_state.mean(dim=1)
                        
                        # Calculate violence indicators from features
                        # Look for high activation patterns that might indicate violence
                        activation_variance = torch.var(features).item()
                        activation_mean = torch.mean(torch.abs(features)).item()
                        
                        # Combine metrics for violence score
                        violence_score = min(activation_variance * 10 + activation_mean * 5, 1.0)
                        feature_scores.append(violence_score)
            
            return np.mean(feature_scores) if feature_scores else 0.0
            
        except Exception as e:
            print(f"Visual feature extraction error: {e}")
            return 0.0
    
    def extract_temporal_features(self, frames_sequence):
        if len(frames_sequence) < 2:
            return np.zeros(10)
        
        # Convert to grayscale sequence
        gray_sequence = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_sequence]
        
        features = []
        
        # 1. Frame difference analysis
        frame_diffs = []
        for i in range(1, len(gray_sequence)):
            diff = cv2.absdiff(gray_sequence[i], gray_sequence[i-1])
            frame_diffs.append(np.mean(diff))
        
        features.extend([
            np.mean(frame_diffs),
            np.std(frame_diffs),
            np.max(frame_diffs),
            len([d for d in frame_diffs if d > np.mean(frame_diffs) + np.std(frame_diffs)])
        ])
        
        # 2. Optical flow temporal consistency
        flow_magnitudes = []
        for i in range(1, len(gray_sequence)):
            flow = cv2.calcOpticalFlowFarneback(
                gray_sequence[i-1], gray_sequence[i], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(np.mean(magnitude))
        
        features.extend([
            np.mean(flow_magnitudes),
            np.std(flow_magnitudes),
            np.max(flow_magnitudes) - np.min(flow_magnitudes)
        ])
        
        # 3. Temporal gradient analysis
        temporal_gradients = []
        for i in range(2, len(gray_sequence)):
            grad = np.abs(gray_sequence[i].astype(float) - 2*gray_sequence[i-1].astype(float) + gray_sequence[i-2].astype(float))
            temporal_gradients.append(np.mean(grad))
        
        features.extend([
            np.mean(temporal_gradients),
            np.std(temporal_gradients),
            np.max(temporal_gradients)
        ])
        
        return np.array(features)
    
    def extract_spatial_violence_patterns(self, frame, person_boxes):
        features = []
        
        # 1. Person arrangement analysis
        if len(person_boxes) > 1:
            centers = []
            for box in person_boxes:
                x1, y1, x2, y2 = box
                centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
            
            # Clustering of people (fights often have people clustered)
            if len(centers) >= 2:
                clustering = DBSCAN(eps=100, min_samples=2).fit(centers)
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                features.append(n_clusters / len(centers))  # Cluster density
            else:
                features.append(0)
            
            # Spatial distribution entropy
            distances = []
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                    distances.append(dist)
            
            if distances:
                features.append(np.std(distances) / (np.mean(distances) + 1e-6))
            else:
                features.append(0)
        else:
            features.extend([0, 0])
        
        # 2. Edge and texture analysis in person regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if person_boxes:
            person_textures = []
            person_edges = []
            
            for box in person_boxes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    person_region = gray[y1:y2, x1:x2]
                    
                    # Texture analysis (violence often creates irregular textures)
                    laplacian_var = cv2.Laplacian(person_region, cv2.CV_64F).var()
                    person_textures.append(laplacian_var)
                    
                    # Edge density (violence often has more edges due to motion blur)
                    edges = cv2.Canny(person_region, 50, 150)
                    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                    person_edges.append(edge_density)
            
            features.extend([
                np.mean(person_textures) if person_textures else 0,
                np.std(person_textures) if person_textures else 0,
                np.mean(person_edges) if person_edges else 0,
                np.max(person_edges) if person_edges else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 3. Global motion patterns
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_var = np.var(hsv[:, :, 1])
        value_var = np.var(hsv[:, :, 2])
        
        norm_params = Config.VIOLENCE_DETECTION["feature_normalization"]
        features.extend([
            saturation_var / norm_params["saturation_var_divisor"], 
            value_var / norm_params["value_var_divisor"]
        ])
        
        return np.array(features)

    def analyze_lstm_confidence_distribution(self, lstm_probs):
        
        # Entropy of probability distribution
        entropy = -np.sum(lstm_probs * np.log(lstm_probs + 1e-10))
        
        # Confidence based on max probability
        max_confidence = np.max(lstm_probs)
        
        # Separation between top two classes
        sorted_probs = np.sort(lstm_probs)[::-1]
        separation = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
        return {
            'entropy': entropy,
            'max_confidence': max_confidence,
            'separation': separation,
            'confidence_score': separation * max_confidence  # Combined confidence
        }
    
    async def detect_violence(self, raw_frames: List[np.ndarray], lstm_frames: List[np.ndarray], multi_scale_frames: Dict[float, List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Main violence detection function with multi-modal analysis
        
        Args:
            raw_frames: List of raw video frames for person detection and feature extraction
            lstm_frames: List of preprocessed frames (64x64, normalized) for LSTM
            multi_scale_frames: Optional pre-computed multi-scale frames for robust detection
            
        Returns:
            Dictionary containing detection results and analysis
        """
        logger.info(f"🧠 Violence detection: {len(raw_frames)} frames")
        
        person_tracker = PersonTracker()
        
        # Enhanced data structures
        temporal_features_history = []
        spatial_features_history = []
        person_tracking_history = []
        
        # Multi-scale analysis - use provided multi-scale frames or default scales
        if multi_scale_frames is None:
            scales = [1.0]  # Fallback to single scale if not provided
        else:
            scales = list(multi_scale_frames.keys())
        
        for frame_num, frame in enumerate(raw_frames):
            
            # Multi-scale person detection for robustness
            all_detections = []
            detection_confidences = []
            
            for scale in scales:
                # Use pre-processed scaled frames if available
                if multi_scale_frames and scale in multi_scale_frames:
                    scaled_frame = multi_scale_frames[scale][frame_num] if frame_num < len(multi_scale_frames[scale]) else frame
                else:
                    scaled_frame = frame  # Fallback to original frame
                
                results = self.yolo_model(scaled_frame, verbose=False)
                scale_detections = []
                scale_confidences = []
                
                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.25:  # Lower threshold
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Scale back to original frame size
                            if scale != 1.0:
                                x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                            
                            scale_detections.append((x1, y1, x2, y2))
                            scale_confidences.append(float(box.conf[0]))
                
                all_detections.extend(scale_detections)
                detection_confidences.extend(scale_confidences)
            
            # Non-maximum suppression across scales
            if all_detections:
                indices = cv2.dnn.NMSBoxes(
                    all_detections, detection_confidences, 0.25, 0.4
                )
                
                if len(indices) > 0:
                    final_detections = [all_detections[i] for i in indices.flatten()]
                    final_confidences = [detection_confidences[i] for i in indices.flatten()]
                else:
                    final_detections = all_detections[:1]  # Keep at least one
                    final_confidences = detection_confidences[:1]
            else:
                final_detections = []
                final_confidences = []
            
            # Update tracking
            person_tracker.update(final_detections)
            
            # Extract spatial features
            spatial_features = self.extract_spatial_violence_patterns(frame, final_detections)
            spatial_features_history.append(spatial_features)
            
            # Track person dynamics
            person_tracking_history.append({
                'count': len(final_detections),
                'avg_confidence': np.mean(final_confidences) if final_confidences else 0,
                'max_confidence': np.max(final_confidences) if final_confidences else 0,
                'detections': final_detections
            })
        
        # Extract temporal features from frame sequence
        if len(raw_frames) >= 2:
            temporal_features = self.extract_temporal_features(raw_frames)
            temporal_features_history.append(temporal_features)
        
        # LSTM Ensemble Prediction 
        lstm_predictions = []
        lstm_confidences = []
        
        if len(lstm_frames) == self.sequence_length:
            # Multiple LSTM inferences with slight variations for ensemble
            for _ in range(3):  # 3 different inferences
                try:
                    lstm_probs = self.mobilstm_model.predict(
                        np.expand_dims(lstm_frames, axis=0), verbose=0
                    )[0]
                    
                    confidence_analysis = self.analyze_lstm_confidence_distribution(lstm_probs)
                    
                    lstm_predictions.append(lstm_probs[1])
                    lstm_confidences.append(confidence_analysis)
                    
                except Exception as e:
                    lstm_predictions.append(0.0)
                    lstm_confidences.append({'confidence_score': 0.0})
        
        # Aggregate LSTM results
        if lstm_predictions:
            lstm_prediction = np.mean(lstm_predictions)
            lstm_std = np.std(lstm_predictions)
            avg_confidence = np.mean([c['confidence_score'] for c in lstm_confidences])
        else:
            lstm_prediction = 0.0
            lstm_std = 0.0
            avg_confidence = 0.0
        
        # Advanced feature aggregation
        # 1. Temporal features
        temporal_score = 0.0
        if temporal_features_history:
            temporal_features_agg = np.mean(temporal_features_history, axis=0)
            
            # Normalize and weight temporal features using config
            norm_params = Config.VIOLENCE_DETECTION["feature_normalization"]
            motion_intensity = temporal_features_agg[0] / norm_params["motion_intensity_divisor"]
            motion_variability = temporal_features_agg[1] / norm_params["motion_variability_divisor"]
            motion_peaks = temporal_features_agg[3] / norm_params["motion_peaks_divisor"]
            
            temporal_score = min((motion_intensity * 0.4 + motion_variability * 0.4 + motion_peaks * 0.2), 1.0)
        
        # 2. Spatial features
        spatial_score = 0.0
        if spatial_features_history:
            spatial_features_agg = np.mean(spatial_features_history, axis=0)
            
            # Normalize spatial features using config
            norm_params = Config.VIOLENCE_DETECTION["feature_normalization"]
            clustering_score = spatial_features_agg[0]  # Already normalized
            spatial_distribution = min(spatial_features_agg[1] / norm_params["spatial_distribution_divisor"], 1.0)
            texture_intensity = min(spatial_features_agg[2] / norm_params["texture_intensity_divisor"], 1.0)
            edge_density = min(spatial_features_agg[4] * norm_params["edge_density_multiplier"], 1.0)
            
            spatial_score = (clustering_score * 0.3 + spatial_distribution * 0.2 + 
                            texture_intensity * 0.3 + edge_density * 0.2)
        
        # 3. Person interaction analysis
        interaction_score = 0.0
        if person_tracking_history:
            avg_people = np.mean([p['count'] for p in person_tracking_history])
            avg_detection_confidence = np.mean([p['avg_confidence'] for p in person_tracking_history])
            people_variability = np.std([p['count'] for p in person_tracking_history])
            
            # Multiple people with high confidence and variability suggests action
            interaction_score = min((
                (avg_people / 4.0) * 0.4 +
                avg_detection_confidence * 0.3 +
                (people_variability / 2.0) * 0.3
            ), 1.0)
        
        # 4. Visual features (HuggingFace)
        visual_score = 0.0
        if self.hf_model_available and len(lstm_frames) > 0:
            visual_score = self.extract_visual_features(lstm_frames)
        
        # Advanced ensemble with adaptive weighting
        # Base weights from configuration
        base_weights = Config.VIOLENCE_DETECTION["ensemble_weights"].copy()
        
        # Adaptive weight adjustment based on confidence
        confidence_factor = avg_confidence
        uncertainty_penalty = lstm_std  # Penalize if LSTM is uncertain
        
        # Get thresholds from config
        conf_thresholds = Config.VIOLENCE_DETECTION["confidence_thresholds"]
        weight_adjustments = Config.VIOLENCE_DETECTION["weight_adjustments"]
        
        # Boost LSTM weight if highly confident
        if confidence_factor > conf_thresholds["high_confidence_min"] and uncertainty_penalty < conf_thresholds["low_uncertainty_max"]:
            base_weights['lstm'] = min(weight_adjustments["lstm_max_boost"], base_weights['lstm'] + weight_adjustments["lstm_boost_amount"])
            # Redistribute remaining weight
            remaining = 1.0 - base_weights['lstm']
            for key in ['temporal', 'spatial', 'interaction', 'visual']:
                base_weights[key] = base_weights[key] * (remaining / weight_adjustments["remaining_weight_base"])
        
        # Reduce LSTM weight if uncertain, boost other features
        elif uncertainty_penalty > conf_thresholds["uncertainty_penalty_threshold"] or confidence_factor < conf_thresholds["low_confidence_max"]:
            base_weights['lstm'] = max(weight_adjustments["lstm_min_reduction"], base_weights['lstm'] - weight_adjustments["lstm_reduction_amount"])
            boost = (0.55 - base_weights['lstm']) / 4
            for key in ['temporal', 'spatial', 'interaction', 'visual']:
                base_weights[key] += boost
        
        # Final ensemble score
        final_score = (
            lstm_prediction * base_weights['lstm'] +
            temporal_score * base_weights['temporal'] +
            spatial_score * base_weights['spatial'] +
            interaction_score * base_weights['interaction'] +
            visual_score * base_weights['visual']
        )
        
        # Advanced adaptive thresholding
        base_threshold = Config.VIOLENCE_DETECTION["base_threshold"]
        
        # Feature agreement analysis
        feature_scores = [lstm_prediction, temporal_score, spatial_score, interaction_score]
        high_scores = sum([1 for score in feature_scores if score > 0.4])
        medium_scores = sum([1 for score in feature_scores if 0.2 < score <= 0.4])
        
        # Get threshold adjustment factors from config
        threshold_adj = Config.VIOLENCE_DETECTION["threshold_adjustments"]
        conf_thresholds = Config.VIOLENCE_DETECTION["confidence_thresholds"]
        
        # Adjust threshold based on feature agreement and confidence
        if high_scores >= 3:  
            threshold = base_threshold * threshold_adj["strong_agreement_factor"]
        elif high_scores >= 2 and medium_scores >= 1:  
            threshold = base_threshold * threshold_adj["good_agreement_factor"]
        elif confidence_factor > 0.6: 
            threshold = base_threshold * threshold_adj["high_confidence_factor"]
        elif uncertainty_penalty > 0.25: 
            threshold = base_threshold * threshold_adj["high_uncertainty_factor"]
        else:
            threshold = base_threshold
        
        # Final confidence boosting for very strong signals
        if (lstm_prediction > conf_thresholds["lstm_boost_threshold"] and 
            temporal_score > conf_thresholds["temporal_boost_threshold"] and 
            confidence_factor > conf_thresholds["high_confidence_min"]):
            final_score = min(final_score * threshold_adj["confidence_boost_factor"], 1.0)
            threshold = threshold * threshold_adj["confidence_boost_threshold_factor"]
        
        # Violence detection decision
        is_violence = final_score > threshold
        
        # Additional validation: If borderline, require multiple indicators
        borderline_min = conf_thresholds["borderline_min"]
        borderline_max = conf_thresholds["borderline_max"]
        if borderline_min < final_score < borderline_max:
            strong_indicators = sum([
                lstm_prediction > 0.3,
                temporal_score > 0.4,
                spatial_score > 0.3,
                interaction_score > 0.3
            ])
            if strong_indicators < 2:
                is_violence = False
        
        result = {
            'is_violence': is_violence,
            'confidence': final_score,
            'lstm_score': lstm_prediction,
            'lstm_confidence': avg_confidence,
            'lstm_uncertainty': uncertainty_penalty,
            'temporal_score': temporal_score,
            'spatial_score': spatial_score,
            'interaction_score': interaction_score,
            'visual_score': visual_score,
            'threshold': threshold,
            'feature_agreement': high_scores,
            'weights': base_weights,
            'ensemble_components': {
                'lstm_predictions': lstm_predictions,
                'temporal_features': temporal_features_history,
                'spatial_features': len(spatial_features_history),
                'person_tracking': len(person_tracking_history)
            }
        }
        
        logger.info(f"🧠 Violence: {'⚠️ DETECTED' if is_violence else '✅ SAFE'} (confidence: {final_score:.1%})")
        return result