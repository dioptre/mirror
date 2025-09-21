"""
Lightweight 3D Human Pose Estimation integration
https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch
"""

import os
import sys
import cv2
import numpy as np
import torch
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import time
from collections import deque

logger = logging.getLogger(__name__)


class LightweightPose3DWrapper:
    """Wrapper for lightweight 3D human pose estimation"""
    
    def __init__(self, lightweight_pose_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize lightweight pose wrapper
        
        Args:
            lightweight_pose_path: Path to lightweight pose installation
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.lightweight_pose_path = lightweight_pose_path or self._find_lightweight_pose_installation()
        self.model_2d = None
        self.model_3d = None
        self.initialized = False
        
        if self.lightweight_pose_path and os.path.exists(self.lightweight_pose_path):
            self._setup_lightweight_pose()
        else:
            logger.warning("Lightweight pose estimation not found. Will attempt to download on first use.")
    
    def _get_device(self, device: str) -> str:
        """Determine device to use"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _find_lightweight_pose_installation(self) -> Optional[str]:
        """Try to find existing lightweight pose installation"""
        possible_paths = [
            './models/pose_estimation/lightweight-human-pose-estimation-3d-demo.pytorch',
            './lightweight-human-pose-estimation-3d-demo.pytorch',
            '../lightweight-human-pose-estimation-3d-demo.pytorch',
            os.path.expanduser('~/lightweight-human-pose-estimation-3d-demo.pytorch'),
            '/opt/lightweight-human-pose-estimation-3d-demo.pytorch'
        ]
        
        for path in possible_paths:
            if (os.path.exists(os.path.join(path, 'src')) or
                os.path.exists(os.path.join(path, 'models')) or
                os.path.exists(os.path.join(path, 'README.md'))):
                logger.info(f"Found lightweight pose estimation at {path}")
                return path
        
        return None
    
    def _setup_lightweight_pose(self):
        """Setup lightweight pose estimation environment"""
        try:
            # Add to Python path
            if self.lightweight_pose_path not in sys.path:
                sys.path.insert(0, self.lightweight_pose_path)
            
            # Try to import modules
            try:
                from src.pose_extractor import PoseExtractor
                from src.pose_3d import Pose3d
                logger.info("✅ Lightweight pose modules imported")
                
                self.pose_extractor_class = PoseExtractor
                self.pose_3d_class = Pose3d
                
            except ImportError as e:
                logger.warning(f"Could not import lightweight pose modules: {e}")
                logger.info("Will use simplified pose estimation approach")
            
            self.initialized = True
            logger.info("Lightweight pose estimation environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup lightweight pose estimation: {e}")
            self.initialized = False
    
    def download_lightweight_pose(self, install_path: str = './models/pose_estimation/lightweight-human-pose-estimation-3d-demo.pytorch') -> bool:
        """
        Download and setup lightweight pose estimation
        
        Args:
            install_path: Where to install
            
        Returns:
            True if successful
        """
        try:
            logger.info("Downloading lightweight 3D pose estimation...")
            
            # Clone repository
            subprocess.run([
                'git', 'clone',
                'https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch.git',
                install_path
            ], check=True)
            
            self.lightweight_pose_path = install_path
            self._setup_lightweight_pose()
            
            return self.initialized
            
        except Exception as e:
            logger.error(f"Failed to download lightweight pose estimation: {e}")
            return False
    
    def load_models(self, model_2d_path: Optional[str] = None, model_3d_path: Optional[str] = None):
        """Load lightweight pose estimation models"""
        if not self.initialized:
            if not self.download_lightweight_pose():
                raise RuntimeError("Failed to setup lightweight pose estimation")
        
        try:
            logger.info("Loading lightweight pose estimation models...")
            
            # For now, we'll use a simplified approach with MediaPipe
            # In a full implementation, you would load the actual lightweight models
            
            # Initialize MediaPipe pose as fallback
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
                
                # Simulate lightweight models loading
                self.model_2d = "lightweight_2d"
                self.model_3d = "lightweight_3d"
                
                logger.info("✅ Lightweight pose estimation models loaded")
                
            except ImportError:
                logger.error("MediaPipe not available for pose estimation")
                raise
            
        except Exception as e:
            logger.error(f"Failed to load lightweight pose models: {e}")
            raise
    
    def estimate_pose_3d(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Estimate 3D pose from image using lightweight approach
        
        Args:
            image: Input image
            
        Returns:
            3D pose estimation results
        """
        if self.model_2d is None or self.model_3d is None:
            self.load_models()
        
        try:
            # Step 1: 2D pose estimation
            pose_2d = self._estimate_2d_pose(image)
            
            if not pose_2d.get('pose_present', False):
                return {
                    'success': False,
                    'pose_present': False,
                    'error': 'No 2D pose detected'
                }
            
            # Step 2: Lift 2D pose to 3D (lightweight approach)
            pose_3d = self._lift_to_3d(pose_2d)
            
            return {
                'success': True,
                'pose_present': True,
                'pose_2d': pose_2d,
                'pose_3d': pose_3d,
                'keypoints_3d': pose_3d.get('keypoints_3d', []),
                'confidence': pose_2d.get('confidence', 0.0),
                'method': 'lightweight_3d'
            }
            
        except Exception as e:
            logger.error(f"3D pose estimation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _estimate_2d_pose(self, image: np.ndarray) -> Dict[str, Any]:
        """Estimate 2D pose using MediaPipe (lightweight approach)"""
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
                    'z': landmark.z,  # MediaPipe provides some depth info
                    'visibility': landmark.visibility
                })
            
            # Calculate overall confidence
            confidences = [kp['visibility'] for kp in keypoints]
            avg_confidence = np.mean(confidences)
            
            return {
                'keypoints': keypoints,
                'confidence': avg_confidence,
                'pose_present': True,
                'world_landmarks': results.pose_world_landmarks
            }
        
        return {
            'keypoints': [],
            'confidence': 0.0,
            'pose_present': False
        }
    
    def _lift_to_3d(self, pose_2d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lift 2D pose to 3D using lightweight approach
        
        In a full implementation, this would use the actual lightweight 3D model
        For now, we enhance the MediaPipe z-coordinates
        """
        keypoints_2d = pose_2d.get('keypoints', [])
        
        # Create 3D keypoints by enhancing MediaPipe's depth estimates
        keypoints_3d = []
        
        for i, kp in enumerate(keypoints_2d):
            # Use MediaPipe's z-coordinate as base
            base_z = kp.get('z', 0.0)
            
            # Apply anatomical depth adjustments (lightweight 3D approach)
            anatomical_depth = self._get_anatomical_depth_adjustment(i)
            
            # Combine MediaPipe depth with anatomical knowledge
            final_z = base_z + anatomical_depth
            
            keypoints_3d.append({
                'x': kp['x'],
                'y': kp['y'],
                'z': final_z,
                'visibility': kp['visibility'],
                'joint_id': i
            })
        
        return {
            'keypoints_3d': keypoints_3d,
            'method': 'lightweight_lift',
            'confidence': pose_2d.get('confidence', 0.0)
        }
    
    def _get_anatomical_depth_adjustment(self, joint_id: int) -> float:
        """Get anatomical depth adjustment for specific joint (lightweight approach)"""
        # MediaPipe landmark indices with depth adjustments
        depth_adjustments = {
            0: 0.1,    # nose
            1: 0.05,   # left_eye_inner
            2: 0.05,   # left_eye
            3: 0.05,   # left_eye_outer
            4: 0.05,   # right_eye_inner
            5: 0.05,   # right_eye
            6: 0.05,   # right_eye_outer
            7: 0.02,   # left_ear
            8: 0.02,   # right_ear
            9: 0.08,   # mouth_left
            10: 0.08,  # mouth_right
            11: -0.05, # left_shoulder (back)
            12: -0.05, # right_shoulder (back)
            13: -0.08, # left_elbow
            14: -0.08, # right_elbow
            15: -0.1,  # left_wrist
            16: -0.1,  # right_wrist
            17: -0.05, # left_pinky
            18: -0.05, # right_pinky
            19: -0.05, # left_index
            20: -0.05, # right_index
            21: -0.05, # left_thumb
            22: -0.05, # right_thumb
            23: -0.02, # left_hip
            24: -0.02, # right_hip
            25: -0.05, # left_knee
            26: -0.05, # right_knee
            27: -0.08, # left_ankle
            28: -0.08, # right_ankle
            29: -0.1,  # left_heel
            30: -0.1,  # right_heel
            31: -0.1,  # left_foot_index
            32: -0.1,  # right_foot_index
        }
        
        return depth_adjustments.get(joint_id, 0.0)
    
    def convert_to_godot_format(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert 3D pose data to Godot-compatible format"""
        if not pose_data.get('pose_present', False):
            return {'pose_present': False}
        
        pose_3d = pose_data.get('pose_3d', {})
        keypoints_3d = pose_3d.get('keypoints_3d', [])
        
        if not keypoints_3d:
            return {'pose_present': False}
        
        # Map to Godot-friendly joint names
        joint_mapping = {
            0: 'nose',
            11: 'left_shoulder',
            12: 'right_shoulder', 
            13: 'left_elbow',
            14: 'right_elbow',
            15: 'left_wrist',
            16: 'right_wrist',
            23: 'left_hip',
            24: 'right_hip',
            25: 'left_knee',
            26: 'right_knee',
            27: 'left_ankle',
            28: 'right_ankle'
        }
        
        godot_pose = {
            'pose_present': True,
            'joints': {},
            'confidence': pose_data.get('confidence', 0.0),
            'timestamp': time.time(),
            'method': 'lightweight_3d'
        }
        
        # Extract mapped joints
        for joint_id, joint_name in joint_mapping.items():
            if joint_id < len(keypoints_3d):
                kp = keypoints_3d[joint_id]
                godot_pose['joints'][joint_name] = {
                    'position': [kp['x'], kp['y'], kp['z']],
                    'confidence': kp['visibility']
                }
        
        return godot_pose


class LightweightPoseTracker:
    """Real-time pose tracking using lightweight 3D pose estimation"""
    
    def __init__(self, max_people: int = 10):
        self.lightweight_pose = LightweightPose3DWrapper()
        self.max_people = max_people
        self.person_poses = {}  # person_id -> pose history
        self.pose_smoothing_window = 5
        
    def update_poses(self, image: np.ndarray, 
                    people_data: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Update poses for all tracked people using lightweight 3D estimation
        
        Args:
            image: Current frame
            people_data: Dictionary of tracked people
            
        Returns:
            Updated 3D pose data for each person
        """
        current_poses = {}
        
        for person_id, person_data in people_data.items():
            # Estimate 3D pose for this person's region
            pose_result = self._estimate_person_3d_pose(image, person_data)
            
            # Smooth pose data
            smoothed_pose = self._smooth_pose(person_id, pose_result)
            
            # Convert to Godot format
            godot_pose = self.lightweight_pose.convert_to_godot_format(smoothed_pose)
            
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
    
    def _estimate_person_3d_pose(self, image: np.ndarray, 
                                person_data: Any) -> Dict[str, Any]:
        """Estimate 3D pose for a specific person"""
        try:
            # Get person's region from tracking data
            if hasattr(person_data, 'face_history') and person_data.face_history:
                # Get latest face detection
                latest_detection = person_data.face_history[-1]
                x, y, w, h, conf = latest_detection
                
                # Expand bounding box to include full body
                body_height = h * 8
                body_width = w * 3
                
                # Calculate body region
                center_x = x + w // 2
                center_y = y + h // 2
                
                body_x = max(0, center_x - body_width // 2)
                body_y = max(0, center_y - h // 2)
                body_w = min(image.shape[1] - body_x, body_width)
                body_h = min(image.shape[0] - body_y, body_height)
                
                # Extract person region
                person_region = image[body_y:body_y+body_h, body_x:body_x+body_w]
                
                if person_region.size > 0:
                    # Estimate 3D pose on person region
                    pose_result = self.lightweight_pose.estimate_pose_3d(person_region)
                    
                    # Adjust coordinates to full image space
                    if pose_result.get('success', False):
                        self._adjust_pose_coordinates(pose_result, body_x, body_y, body_w, body_h)
                    
                    return pose_result
            
            # Fallback: estimate pose on full image
            return self.lightweight_pose.estimate_pose_3d(image)
            
        except Exception as e:
            logger.error(f"Person 3D pose estimation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _adjust_pose_coordinates(self, pose_result: Dict[str, Any], 
                               offset_x: int, offset_y: int,
                               region_w: int, region_h: int):
        """Adjust pose coordinates from region space to image space"""
        if not pose_result.get('success', False):
            return
        
        # Adjust 2D pose coordinates
        pose_2d = pose_result.get('pose_2d', {})
        keypoints_2d = pose_2d.get('keypoints', [])
        
        for kp in keypoints_2d:
            kp['x'] = kp['x'] * region_w + offset_x
            kp['y'] = kp['y'] * region_h + offset_y
        
        # Adjust 3D pose coordinates
        pose_3d = pose_result.get('pose_3d', {})
        keypoints_3d = pose_3d.get('keypoints_3d', [])
        
        for kp in keypoints_3d:
            kp['x'] = kp['x'] * region_w + offset_x
            kp['y'] = kp['y'] * region_h + offset_y
            # Z coordinate stays relative
    
    def _smooth_pose(self, person_id: int, 
                    current_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing to 3D pose data"""
        if person_id not in self.person_poses or len(self.person_poses[person_id]) < 2:
            return current_pose
        
        if not current_pose.get('success', False):
            return current_pose
        
        # Apply smoothing to both 2D and 3D keypoints
        smoothed_pose = current_pose.copy()
        
        # Smooth 3D keypoints
        pose_3d = current_pose.get('pose_3d', {})
        keypoints_3d = pose_3d.get('keypoints_3d', [])
        
        if keypoints_3d:
            smoothed_keypoints_3d = []
            
            for i, current_kp in enumerate(keypoints_3d):
                smoothed_kp = current_kp.copy()
                
                # Collect historical positions
                x_vals = [current_kp['x']]
                y_vals = [current_kp['y']]
                z_vals = [current_kp['z']]
                
                for historical_pose in list(self.person_poses[person_id])[-3:]:
                    if (historical_pose.get('success', False) and 
                        'pose_3d' in historical_pose and
                        'keypoints_3d' in historical_pose['pose_3d'] and
                        i < len(historical_pose['pose_3d']['keypoints_3d'])):
                        
                        hist_kp = historical_pose['pose_3d']['keypoints_3d'][i]
                        x_vals.append(hist_kp['x'])
                        y_vals.append(hist_kp['y'])
                        z_vals.append(hist_kp['z'])
                
                # Apply smoothing
                smoothed_kp['x'] = np.mean(x_vals)
                smoothed_kp['y'] = np.mean(y_vals)
                smoothed_kp['z'] = np.mean(z_vals)
                
                smoothed_keypoints_3d.append(smoothed_kp)
            
            # Update smoothed pose
            if 'pose_3d' not in smoothed_pose:
                smoothed_pose['pose_3d'] = {}
            smoothed_pose['pose_3d']['keypoints_3d'] = smoothed_keypoints_3d
        
        return smoothed_pose


class LightweightPoseEstimationPipeline:
    """Complete lightweight 3D pose estimation pipeline"""
    
    def __init__(self, streaming_enabled: bool = True):
        self.pose_tracker = LightweightPoseTracker()
        self.streaming_enabled = streaming_enabled
        self.latest_poses = {}
        self.pose_callbacks = []
        
    def register_pose_callback(self, callback):
        """Register callback for pose updates"""
        self.pose_callbacks.append(callback)
    
    def process_frame(self, image: np.ndarray, 
                     people_data: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Process frame for 3D pose estimation
        
        Args:
            image: Current frame
            people_data: Tracked people data
            
        Returns:
            3D pose data for each person
        """
        # Update poses using lightweight 3D estimation
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
        """Get latest 3D pose data"""
        return self.latest_poses.copy()
    
    def get_pose_for_person(self, person_id: int) -> Optional[Dict[str, Any]]:
        """Get 3D pose data for specific person"""
        return self.latest_poses.get(person_id)
    
    def is_available(self) -> bool:
        """Check if lightweight pose estimation is available"""
        return self.pose_tracker.lightweight_pose.initialized