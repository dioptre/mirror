import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UltraLightFaceDetector:
    """Ultra-Light face detector implementation based on the 1MB model"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.device = self._get_device(device)
        self.model = None
        self.input_size = (320, 240)  # Standard input size for ultra-light model
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.3
        
        if model_path:
            self.load_model(model_path)
    
    def _get_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model(self, model_path: str = None):
        """Load the ultra-light face detection model"""
        self.net = None
        self.cascade = None
        
        # Try to load OpenCV DNN model first
        if model_path:
            try:
                pb_file = os.path.join(model_path, 'opencv_face_detector_uint8.pb')
                pbtxt_file = os.path.join(model_path, 'opencv_face_detector.pbtxt')
                
                if os.path.exists(pb_file) and os.path.exists(pbtxt_file):
                    self.net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
                    logger.info("OpenCV DNN face detection model loaded successfully")
                    return
                else:
                    logger.warning(f"Model files not found at {model_path}")
            except Exception as e:
                logger.warning(f"Could not load DNN model: {e}")
        
        # Try default OpenCV DNN model paths
        try:
            # Look in models directory
            models_dir = './models/face_detection/opencv_face_detector'
            pb_file = os.path.join(models_dir, 'opencv_face_detector_uint8.pb')
            pbtxt_file = os.path.join(models_dir, 'opencv_face_detector.pbtxt')
            
            if os.path.exists(pb_file) and os.path.exists(pbtxt_file):
                self.net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
                logger.info("Face detection model loaded from models directory")
                return
        except Exception as e:
            logger.debug(f"Could not load from models directory: {e}")
        
        # Fallback to Haar cascades
        try:
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("Using Haar cascade fallback for face detection")
        except Exception as e:
            logger.error(f"Failed to load any face detection model: {e}")
            raise RuntimeError("No face detection model could be loaded")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of tuples (x, y, width, height, confidence)
        """
        if self.net is not None:
            return self._detect_with_dnn(image)
        else:
            return self._detect_with_cascade(image)
    
    def _detect_with_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using DNN model"""
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
        
        return faces
    
    def _detect_with_cascade(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Convert to format with confidence (using fixed confidence for cascade)
        return [(x, y, w, h, 0.9) for (x, y, w, h) in faces]
    
    def validate_person_visibility(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> bool:
        """
        Validate that arms, face, and torso are visible
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, width, height)
            
        Returns:
            True if person meets visibility requirements
        """
        x, y, w, h = face_box
        height, width = image.shape[:2]
        
        # Basic heuristics for checking body visibility
        # Face should be in upper portion of image
        face_center_y = y + h // 2
        if face_center_y > height * 0.4:  # Face too low in frame
            return False
        
        # Should have reasonable space below face for torso/arms
        required_body_height = h * 3  # Approximate body height
        available_space = height - (y + h)
        
        if available_space < required_body_height * 0.3:  # Need at least 30% of estimated body (lowered threshold)
            return False
        
        # Face should not be too close to edges (arms likely cut off) - relaxed threshold
        edge_margin = width * 0.05  # Reduced from 0.1 to 0.05
        if x < edge_margin or (x + w) > (width - edge_margin):
            return False
        
        return True


class PersonTracker:
    """Simple person tracker using face detection"""
    
    def __init__(self, max_disappeared: int = 10):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        
    def register(self, centroid: Tuple[int, int]) -> int:
        """Register a new person"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id: int):
        """Remove a tracked person"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> dict:
        """
        Update tracker with new detections
        
        Args:
            detections: List of face bounding boxes
            
        Returns:
            Dictionary mapping person ID to centroid
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # Calculate centroids of detections
        input_centroids = []
        for (x, y, w, h) in detections:
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids.append((cx, cy))
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects to new detections
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Compute distance matrix
            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2
            )
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_rows = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_cols = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # If more objects than detections, mark as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # If more detections than objects, register new ones
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.objects