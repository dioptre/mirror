import os
import sys
import cv2
import numpy as np
import torch
import subprocess
import tempfile
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


class TEMPOWrapper:
    """Wrapper for TEMPO state-of-the-art pose estimation"""
    
    def __init__(self, tempo_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize TEMPO wrapper
        
        Args:
            tempo_path: Path to TEMPO installation
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.tempo_path = tempo_path or self._find_tempo_installation()
        self.model = None
        self.initialized = False
        
        if self.tempo_path and os.path.exists(self.tempo_path):
            self._setup_tempo()
        else:
            logger.warning("TEMPO not found. Will attempt to download on first use.")
    
    def _get_device(self, device: str) -> str:
        """Determine device to use"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _find_tempo_installation(self) -> Optional[str]:
        """Try to find existing TEMPO installation"""
        possible_paths = [
            './tempo',
            '../tempo',
            '../../tempo',
            os.path.expanduser('~/tempo'),
            '/opt/tempo'
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'tempo')):
                logger.info(f"Found TEMPO at {path}")
                return path
        
        return None
    
    def _setup_tempo(self):
        """Setup TEMPO environment"""
        try:
            # Add TEMPO to Python path
            if self.tempo_path not in sys.path:
                sys.path.insert(0, self.tempo_path)
            
            self.initialized = True
            logger.info("TEMPO environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup TEMPO: {e}")
            self.initialized = False
    
    def download_tempo(self, install_path: str = './tempo') -> bool:
        """
        Download and setup TEMPO
        
        Args:
            install_path: Where to install TEMPO
            
        Returns:
            True if successful
        """
        try:
            logger.info("Downloading TEMPO...")
            
            # Clone repository
            subprocess.run([
                'git', 'clone',
                'https://github.com/rccchoudhury/tempo.git',
                install_path
            ], check=True)
            
            self.tempo_path = install_path
            self._setup_tempo()
            
            return self.initialized
            
        except Exception as e:
            logger.error(f"Failed to download TEMPO: {e}")
            return False
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load TEMPO model"""
        if not self.initialized:
            if not self.download_tempo():
                raise RuntimeError("Failed to setup TEMPO")
        
        try:
            # For now, we'll use a simplified pose estimation approach
            # In production, you would load the actual TEMPO model
            
            # Initialize MediaPipe as fallback
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mp_drawing = mp.solutions.drawing_utils
                logger.info("MediaPipe pose estimation initialized as fallback")
                
            except ImportError:
                logger.warning("MediaPipe not available")
            
            self.model = "mediapipe"  # Placeholder
            logger.info("TEMPO model loaded (using MediaPipe fallback)")
            
        except Exception as e:
            logger.error(f"Failed to load TEMPO model: {e}")
            raise
    
    def estimate_pose(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Estimate pose from image
        
        Args:
            image: Input image
            
        Returns:
            Pose estimation results
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Use MediaPipe for pose estimation
            pose_data = self._estimate_with_mediapipe(image)
            
            return {
                'success': True,
                'pose_data': pose_data,
                'keypoints': pose_data.get('keypoints', []),
                'confidence': pose_data.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _estimate_with_mediapipe(self, image: np.ndarray) -> Dict[str, Any]:
        """Estimate pose using MediaPipe"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.pose_detector.process(rgb_image)
        
        if results.pose_landmarks:
            # Extract keypoints
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # Calculate overall confidence
            confidences = [kp['visibility'] for kp in keypoints]
            avg_confidence = np.mean(confidences)
            
            return {
                'keypoints': keypoints,
                'confidence': avg_confidence,
                'world_landmarks': results.pose_world_landmarks,
                'pose_present': True
            }
        
        return {
            'keypoints': [],
            'confidence': 0.0,
            'pose_present': False
        }
    
    def convert_to_godot_format(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert pose data to Godot-compatible format
        
        Args:
            pose_data: Raw pose estimation data
            
        Returns:
            Godot-compatible pose data
        """
        if not pose_data.get('pose_present', False):
            return {'pose_present': False}
        
        keypoints = pose_data.get('keypoints', [])
        
        # MediaPipe landmark indices (subset for key joints)
        joint_mapping = {
            'nose': 0,
            'left_eye': 2,
            'right_eye': 5,
            'left_ear': 7,
            'right_ear': 8,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        godot_pose = {
            'pose_present': True,
            'joints': {},
            'confidence': pose_data.get('confidence', 0.0),
            'timestamp': time.time()
        }
        
        # Extract key joints
        for joint_name, landmark_idx in joint_mapping.items():
            if landmark_idx < len(keypoints):
                kp = keypoints[landmark_idx]
                godot_pose['joints'][joint_name] = {
                    'position': [kp['x'], kp['y'], kp['z']],
                    'confidence': kp['visibility']
                }
        
        return godot_pose


class PoseTracker:
    """Real-time pose tracking for multiple people"""
    
    def __init__(self, max_people: int = 10):
        self.tempo = TEMPOWrapper()
        self.max_people = max_people
        self.person_poses = {}  # person_id -> pose history
        self.pose_smoothing_window = 5
        
    def update_poses(self, image: np.ndarray, 
                    people_data: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Update poses for all tracked people
        
        Args:
            image: Current frame
            people_data: Dictionary of tracked people
            
        Returns:
            Updated pose data for each person
        """
        current_poses = {}
        
        for person_id, person_data in people_data.items():
            # Estimate pose for this person's region
            pose_result = self._estimate_person_pose(image, person_data)
            
            # Smooth pose data
            smoothed_pose = self._smooth_pose(person_id, pose_result)
            
            # Convert to Godot format
            godot_pose = self.tempo.convert_to_godot_format(smoothed_pose)
            
            current_poses[person_id] = godot_pose
            
            # Update history
            if person_id not in self.person_poses:
                self.person_poses[person_id] = deque(maxlen=self.pose_smoothing_window)
            self.person_poses[person_id].append(pose_result)
        
        # Clean up poses for people who disappeared
        active_people = set(people_data.keys())
        disappeared_people = set(self.person_poses.keys()) - active_people
        for person_id in disappeared_people:
            del self.person_poses[person_id]
        
        return current_poses
    
    def _estimate_person_pose(self, image: np.ndarray, 
                            person_data: Any) -> Dict[str, Any]:
        """Estimate pose for a specific person"""
        try:
            # Get person's bounding box from tracking data
            if hasattr(person_data, 'face_history') and person_data.face_history:
                # Get latest face detection
                latest_detection = person_data.face_history[-1]
                x, y, w, h, conf = latest_detection
                
                # Expand bounding box to include full body (estimate)
                body_height = h * 8  # Estimate full body height
                body_width = w * 3   # Estimate body width
                
                # Calculate body region
                center_x = x + w // 2
                center_y = y + h // 2
                
                body_x = max(0, center_x - body_width // 2)
                body_y = max(0, center_y - h // 2)  # Start from face
                body_w = min(image.shape[1] - body_x, body_width)
                body_h = min(image.shape[0] - body_y, body_height)
                
                # Extract person region
                person_region = image[body_y:body_y+body_h, body_x:body_x+body_w]
                
                if person_region.size > 0:
                    # Estimate pose on person region
                    pose_result = self.tempo.estimate_pose(person_region)
                    
                    # Adjust coordinates to full image space
                    if pose_result.get('success', False):
                        self._adjust_pose_coordinates(pose_result, body_x, body_y, body_w, body_h)
                    
                    return pose_result
            
            # Fallback: estimate pose on full image
            return self.tempo.estimate_pose(image)
            
        except Exception as e:
            logger.error(f"Person pose estimation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _adjust_pose_coordinates(self, pose_result: Dict[str, Any], 
                               offset_x: int, offset_y: int,
                               region_w: int, region_h: int):
        """Adjust pose coordinates from region space to image space"""
        if not pose_result.get('success', False):
            return
        
        pose_data = pose_result.get('pose_data', {})
        keypoints = pose_data.get('keypoints', [])
        
        for kp in keypoints:
            # Convert normalized coordinates to region coordinates, then to image coordinates
            kp['x'] = (kp['x'] * region_w + offset_x) / region_w  # Normalize to full image
            kp['y'] = (kp['y'] * region_h + offset_y) / region_h
    
    def _smooth_pose(self, person_id: int, 
                    current_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing to pose data"""
        if person_id not in self.person_poses or len(self.person_poses[person_id]) < 2:
            return current_pose
        
        if not current_pose.get('success', False):
            return current_pose
        
        # Get pose history
        pose_history = list(self.person_poses[person_id])
        
        # Apply simple moving average to keypoints
        current_keypoints = current_pose.get('pose_data', {}).get('keypoints', [])
        
        if not current_keypoints:
            return current_pose
        
        smoothed_keypoints = []
        
        for i, current_kp in enumerate(current_keypoints):
            smoothed_kp = current_kp.copy()
            
            # Collect historical positions for this keypoint
            x_vals = [current_kp['x']]
            y_vals = [current_kp['y']]
            z_vals = [current_kp['z']]
            
            for historical_pose in pose_history[-3:]:  # Use last 3 poses
                if (historical_pose.get('success', False) and 
                    'pose_data' in historical_pose and
                    'keypoints' in historical_pose['pose_data'] and
                    i < len(historical_pose['pose_data']['keypoints'])):
                    
                    hist_kp = historical_pose['pose_data']['keypoints'][i]
                    x_vals.append(hist_kp['x'])
                    y_vals.append(hist_kp['y'])
                    z_vals.append(hist_kp['z'])
            
            # Apply smoothing
            smoothed_kp['x'] = np.mean(x_vals)
            smoothed_kp['y'] = np.mean(y_vals)
            smoothed_kp['z'] = np.mean(z_vals)
            
            smoothed_keypoints.append(smoothed_kp)
        
        # Create smoothed result
        smoothed_result = current_pose.copy()
        smoothed_result['pose_data']['keypoints'] = smoothed_keypoints
        
        return smoothed_result


class PoseEstimationPipeline:
    """Complete pose estimation pipeline"""
    
    def __init__(self, streaming_enabled: bool = True):
        self.pose_tracker = PoseTracker()
        self.streaming_enabled = streaming_enabled
        self.latest_poses = {}
        self.pose_callbacks = []
        
    def register_pose_callback(self, callback):
        """Register callback for pose updates"""
        self.pose_callbacks.append(callback)
    
    def process_frame(self, image: np.ndarray, 
                     people_data: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Process frame for pose estimation
        
        Args:
            image: Current frame
            people_data: Tracked people data
            
        Returns:
            Pose data for each person
        """
        # Update poses
        poses = self.pose_tracker.update_poses(image, people_data)
        
        # Store latest poses
        self.latest_poses = poses
        
        # Notify callbacks
        for callback in self.pose_callbacks:
            try:
                callback(poses)
            except Exception as e:
                logger.error(f"Pose callback failed: {e}")
        
        return poses
    
    def get_latest_poses(self) -> Dict[int, Dict[str, Any]]:
        """Get latest pose data"""
        return self.latest_poses.copy()
    
    def get_pose_for_person(self, person_id: int) -> Optional[Dict[str, Any]]:
        """Get pose data for specific person"""
        return self.latest_poses.get(person_id)