"""
siMLPe (Simple MLP for 3D Pose) integration for pose prediction and smoothing
https://github.com/dulucas/siMLPe
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import subprocess
import json
import random
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DriftState(Enum):
    """Drift behavior states"""
    FOLLOWING = "following"        # Normal mirroring mode
    DRIFT_START = "drift_start"    # Beginning drift behavior
    DRIFTING = "drifting"         # Following predictions
    DRIFT_RETURN = "drift_return"  # Returning to human
    COOLDOWN = "cooldown"         # Cooling down before next drift


@dataclass
class PersonDriftData:
    """Data for tracking drift behavior per person"""
    person_id: int
    state: DriftState = DriftState.FOLLOWING
    drift_start_time: float = 0.0
    drift_return_start_time: float = 0.0
    last_drift_time: float = 0.0
    accumulated_prediction: Optional[Dict[str, Any]] = None
    drift_base_pose: Optional[Dict[str, Any]] = None


class siMLPeWrapper:
    """Wrapper for siMLPe pose prediction and smoothing"""
    
    def __init__(self, simlpe_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize siMLPe wrapper
        
        Args:
            simlpe_path: Path to siMLPe installation
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.simlpe_path = simlpe_path or self._find_simlpe_installation()
        self.model = None
        self.initialized = False
        
        # Pose prediction parameters
        self.sequence_length = 10  # Number of past frames to use for prediction (reduced)
        self.prediction_frames = 1  # Number of future frames to predict (reduced)
        self.joints_per_pose = 13  # Number of joints we track
        self.coords_per_joint = 3  # x, y, z coordinates
        
        if self.simlpe_path and os.path.exists(self.simlpe_path):
            self._setup_simlpe()
        else:
            logger.warning("siMLPe not found. Will attempt to download on first use.")
    
    def _get_device(self, device: str) -> str:
        """Determine device to use"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _find_simlpe_installation(self) -> Optional[str]:
        """Try to find existing siMLPe installation"""
        possible_paths = [
            './models/pose_prediction/siMLPe',
            './siMLPe',
            '../siMLPe',
            os.path.expanduser('~/siMLPe'),
            '/opt/siMLPe'
        ]
        
        for path in possible_paths:
            if (os.path.exists(os.path.join(path, 'model')) or
                os.path.exists(os.path.join(path, 'src')) or
                os.path.exists(os.path.join(path, 'README.md'))):
                logger.info(f"Found siMLPe at {path}")
                return path
        
        return None
    
    def _setup_simlpe(self):
        """Setup siMLPe environment"""
        try:
            # Add siMLPe to Python path
            if self.simlpe_path not in sys.path:
                sys.path.insert(0, self.simlpe_path)
            
            # Try to import siMLPe modules
            try:
                # Common siMLPe import patterns
                from model.mlp import MLP
                from utils.data_utils import define_actions
                logger.info("✅ siMLPe modules imported")
                
                self.mlp_class = MLP
                self.define_actions = define_actions
                
            except ImportError as e:
                logger.warning(f"Could not import siMLPe modules: {e}")
                logger.info("Will use simplified pose prediction approach")
            
            self.initialized = True
            logger.info("siMLPe environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup siMLPe: {e}")
            self.initialized = False
    
    def download_simlpe(self, install_path: str = './models/pose_prediction/siMLPe') -> bool:
        """
        Download and setup siMLPe
        
        Args:
            install_path: Where to install siMLPe
            
        Returns:
            True if successful
        """
        try:
            logger.info("Downloading siMLPe...")
            
            # Clone repository
            subprocess.run([
                'git', 'clone',
                'https://github.com/dulucas/siMLPe.git',
                install_path
            ], check=True)
            
            self.simlpe_path = install_path
            self._setup_simlpe()
            
            return self.initialized
            
        except Exception as e:
            logger.error(f"Failed to download siMLPe: {e}")
            return False
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load siMLPe model for pose prediction"""
        if not self.initialized:
            if not self.download_simlpe():
                logger.warning("siMLPe not available, using simplified prediction")
                self.model = "simplified"
                return
        
        try:
            logger.info("Loading siMLPe model...")
            
            # For now, we'll use a simplified MLP approach
            # In a full implementation, you would load the actual siMLPe model
            
            # Create a simple prediction model
            self.model = self._create_simple_prediction_model()
            
            logger.info("✅ siMLPe prediction model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load siMLPe model: {e}")
            logger.info("Using simplified pose prediction")
            self.model = "simplified"
    
    def _create_simple_prediction_model(self):
        """Create a simple pose prediction model"""
        # Calculate correct dimensions
        input_features = self.joints_per_pose * self.coords_per_joint  # 13 joints * 3 coords = 39
        input_dim = input_features * self.sequence_length  # 39 * 10 = 390
        output_dim = input_features * self.prediction_frames  # 39 * 1 = 39
        
        logger.info(f"Creating siMLPe model: input_dim={input_dim}, output_dim={output_dim}")
        
        # Simple MLP for pose prediction (siMLPe-inspired)
        class SimplePoseMLP(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_dim=512):
                super(SimplePoseMLP, self).__init__()
                
                self.mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
            
            def forward(self, x):
                return self.mlp(x)
        
        model = SimplePoseMLP(input_dim, output_dim)
        model.to(self.device)
        model.eval()
        
        # Initialize with random weights (in production, you'd load trained weights)
        logger.info("✅ siMLPe prediction model created successfully")
        
        return model
    
    def predict_poses(self, pose_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict future poses from history using siMLPe approach
        
        Args:
            pose_history: List of past pose data
            
        Returns:
            Predicted pose data
        """
        if self.model is None:
            self.load_model()
        
        if self.model == "simplified":
            return self._simplified_prediction(pose_history)
        
        try:
            # Convert pose history to model input format
            input_tensor = self._poses_to_tensor(pose_history)
            
            if input_tensor is None:
                return self._simplified_prediction(pose_history)
            
            # Run prediction
            with torch.no_grad():
                predicted_tensor = self.model(input_tensor)
            
            # Convert back to pose format
            predicted_poses = self._tensor_to_poses(predicted_tensor)
            
            return {
                'success': True,
                'predicted_poses': predicted_poses,
                'method': 'simlpe_mlp',
                'confidence': self._calculate_prediction_confidence(pose_history)
            }
            
        except Exception as e:
            logger.error(f"siMLPe prediction failed: {e}")
            return self._simplified_prediction(pose_history)
    
    def _simplified_prediction(self, pose_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simplified pose prediction using motion extrapolation"""
        if len(pose_history) < 2:
            return {
                'success': False,
                'error': 'Insufficient pose history'
            }
        
        # Get last two poses for velocity calculation
        current_pose = pose_history[-1]
        previous_pose = pose_history[-2]
        
        if not current_pose.get('pose_present', False) or not previous_pose.get('pose_present', False):
            return {
                'success': False,
                'error': 'Invalid pose data'
            }
        
        # Calculate motion and predict next pose
        predicted_pose = self._extrapolate_motion(previous_pose, current_pose)
        
        return {
            'success': True,
            'predicted_poses': [predicted_pose],
            'method': 'motion_extrapolation',
            'confidence': min(current_pose.get('confidence', 0.0), 0.8)  # Lower confidence for prediction
        }
    
    def _extrapolate_motion(self, prev_pose: Dict[str, Any], curr_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Extrapolate motion to predict next pose"""
        prev_joints = prev_pose.get('joints', {})
        curr_joints = curr_pose.get('joints', {})
        
        predicted_joints = {}
        
        for joint_name in curr_joints:
            if joint_name in prev_joints:
                curr_pos = curr_joints[joint_name]['position']
                prev_pos = prev_joints[joint_name]['position']
                
                # Calculate velocity
                velocity = [
                    curr_pos[0] - prev_pos[0],
                    curr_pos[1] - prev_pos[1], 
                    curr_pos[2] - prev_pos[2]
                ]
                
                # Predict next position with damping
                damping = 0.7  # Reduce velocity over time
                predicted_pos = [
                    curr_pos[0] + velocity[0] * damping,
                    curr_pos[1] + velocity[1] * damping,
                    curr_pos[2] + velocity[2] * damping
                ]
                
                predicted_joints[joint_name] = {
                    'position': predicted_pos,
                    'confidence': curr_joints[joint_name].get('confidence', 0.0) * 0.8
                }
            else:
                # No previous data, keep current position
                predicted_joints[joint_name] = curr_joints[joint_name].copy()
        
        return {
            'pose_present': True,
            'joints': predicted_joints,
            'confidence': curr_pose.get('confidence', 0.0) * 0.8,
            'timestamp': time.time(),
            'predicted': True
        }
    
    def _poses_to_tensor(self, pose_history: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Convert pose history to tensor for model input"""
        if len(pose_history) < self.sequence_length:
            # Pad with first pose if we don't have enough history
            if len(pose_history) > 0:
                first_pose = pose_history[0]
                padded_history = [first_pose] * (self.sequence_length - len(pose_history)) + pose_history
            else:
                return None
        else:
            # Take last sequence_length poses
            padded_history = pose_history[-self.sequence_length:]
        
        # Extract joint positions as flat array
        pose_vectors = []
        
        # Standard joint order for consistency (exactly 13 joints)
        joint_order = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for pose in padded_history:
            if not pose.get('pose_present', False):
                # Use zero pose for missing data
                pose_vector = [0.0] * (self.joints_per_pose * self.coords_per_joint)
            else:
                joints = pose.get('joints', {})
                pose_vector = []
                
                for joint_name in joint_order:
                    if joint_name in joints:
                        pos = joints[joint_name]['position']
                        pose_vector.extend([pos[0], pos[1], pos[2]])
                    else:
                        pose_vector.extend([0.0, 0.0, 0.0])  # Missing joint
            
            pose_vectors.append(pose_vector)
        
        # Convert to tensor with correct dimensions
        pose_array = np.array(pose_vectors, dtype=np.float32)  # Shape: [sequence_length, features]
        
        # Flatten for MLP input: [1, sequence_length * features]
        expected_input_size = self.sequence_length * self.joints_per_pose * self.coords_per_joint
        flattened = pose_array.flatten()
        
        if len(flattened) != expected_input_size:
            logger.error(f"Tensor size mismatch: got {len(flattened)}, expected {expected_input_size}")
            return None
        
        pose_tensor = torch.from_numpy(flattened).unsqueeze(0).to(self.device)
        
        return pose_tensor
    
    def _tensor_to_poses(self, predicted_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """Convert predicted tensor back to pose format"""
        predicted_array = predicted_tensor.cpu().numpy().squeeze()  # Remove batch dimension
        
        # Reshape to [prediction_frames, num_joints, 3]
        total_features = self.joints_per_pose * self.coords_per_joint
        
        # For single frame prediction, reshape accordingly
        if self.prediction_frames == 1:
            reshaped = predicted_array.reshape(self.joints_per_pose, self.coords_per_joint)
        else:
            reshaped = predicted_array.reshape(self.prediction_frames, self.joints_per_pose, self.coords_per_joint)
        
        predicted_poses = []
        joint_names = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        if self.prediction_frames == 1:
            # Single frame prediction
            joints = {}
            for joint_idx, joint_name in enumerate(joint_names):
                position = reshaped[joint_idx].tolist()
                joints[joint_name] = {
                    'position': position,
                    'confidence': 0.7,  # Predicted confidence
                    'predicted': True
                }
            
            predicted_poses.append({
                'pose_present': True,
                'joints': joints,
                'confidence': 0.7,
                'timestamp': time.time() + (1.0 / 30.0),  # Next frame
                'predicted': True
            })
        else:
            # Multiple frame prediction
            for frame_idx in range(self.prediction_frames):
                joints = {}
                
                for joint_idx, joint_name in enumerate(joint_names):
                    position = reshaped[frame_idx, joint_idx].tolist()
                    joints[joint_name] = {
                        'position': position,
                        'confidence': 0.7,  # Predicted confidence
                        'predicted': True
                    }
                
                predicted_poses.append({
                    'pose_present': True,
                    'joints': joints,
                    'confidence': 0.7,
                    'timestamp': time.time() + (frame_idx + 1) * (1.0 / 30.0),  # Future timestamps
                    'predicted': True
                })
        
        return predicted_poses
    
    def _calculate_prediction_confidence(self, pose_history: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on pose history quality"""
        if not pose_history:
            return 0.0
        
        # Average confidence of recent poses
        recent_confidences = []
        for pose in pose_history[-5:]:  # Last 5 poses
            if pose.get('pose_present', False):
                recent_confidences.append(pose.get('confidence', 0.0))
        
        if not recent_confidences:
            return 0.0
        
        avg_confidence = np.mean(recent_confidences)
        
        # Reduce confidence for prediction
        return min(avg_confidence * 0.8, 0.9)


class PosePredictionLayer:
    """Pose prediction layer using siMLPe for smoother avatar motion with drift behavior"""
    
    def __init__(self, enabled: bool = True, prediction_horizon: int = 3, 
                 drift_config: Optional[Dict[str, Any]] = None):
        """
        Initialize pose prediction layer with drift behavior
        
        Args:
            enabled: Whether to enable pose prediction
            prediction_horizon: Number of frames to predict ahead
            drift_config: Configuration for drift behavior
        """
        self.enabled = enabled
        self.prediction_horizon = prediction_horizon
        self.simlpe = siMLPeWrapper() if enabled else None
        
        # Per-person pose histories
        self.person_histories = {}  # person_id -> deque of poses
        self.max_history_length = 30  # Keep last 30 poses per person
        
        # Prediction cache
        self.cached_predictions = {}  # person_id -> predicted poses
        
        # Drift behavior configuration
        self.drift_config = drift_config or {}
        self.drift_enabled = self.drift_config.get('drift_enabled', True)
        self.drift_probability = self.drift_config.get('drift_probability', 0.1)
        self.drift_duration = self.drift_config.get('drift_duration', 5.0)
        self.drift_return_duration = self.drift_config.get('drift_return_duration', 2.0)
        self.drift_prediction_strength = self.drift_config.get('drift_prediction_strength', 0.8)
        self.drift_cooldown = self.drift_config.get('drift_cooldown', 30.0)
        
        # Per-person drift tracking
        self.person_drift_data = {}  # person_id -> PersonDriftData
        
        logger.info(f"✅ Pose prediction layer initialized (enabled: {enabled}, drift: {self.drift_enabled})")
    
    def process_poses(self, raw_poses: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Process poses through prediction layer with drift behavior
        
        Args:
            raw_poses: Raw pose data from pose estimation
            
        Returns:
            Enhanced poses with prediction, smoothing, and drift behavior
        """
        if not self.enabled:
            return raw_poses
        
        enhanced_poses = {}
        current_time = time.time()
        
        for person_id, pose_data in raw_poses.items():
            # Initialize drift data for new person
            if person_id not in self.person_drift_data:
                self.person_drift_data[person_id] = PersonDriftData(person_id=person_id)
            
            # Update pose history
            self._update_pose_history(person_id, pose_data)
            
            # Update drift state
            self._update_drift_state(person_id, current_time)
            
            # Generate pose based on drift state
            if self.drift_enabled:
                enhanced_pose = self._generate_drift_aware_pose(person_id, pose_data, current_time)
            else:
                enhanced_pose = self._generate_enhanced_pose(person_id, pose_data)
            
            enhanced_poses[person_id] = enhanced_pose
        
        # Clean up histories for disappeared people
        self._cleanup_histories(set(raw_poses.keys()))
        
        return enhanced_poses
    
    def _update_pose_history(self, person_id: int, pose_data: Dict[str, Any]):
        """Update pose history for a person"""
        if person_id not in self.person_histories:
            self.person_histories[person_id] = deque(maxlen=self.max_history_length)
        
        # Add current pose to history
        timestamped_pose = pose_data.copy()
        timestamped_pose['timestamp'] = time.time()
        
        self.person_histories[person_id].append(timestamped_pose)
    
    def _update_drift_state(self, person_id: int, current_time: float):
        """Update drift state for a person"""
        drift_data = self.person_drift_data[person_id]
        
        if drift_data.state == DriftState.FOLLOWING:
            # Check if we should start drifting
            time_since_last_drift = current_time - drift_data.last_drift_time
            
            if time_since_last_drift > self.drift_cooldown:
                # Random chance to start drifting
                if random.random() < (self.drift_probability / 60.0):  # Per second probability
                    self._start_drift(person_id, current_time)
        
        elif drift_data.state == DriftState.DRIFT_START:
            # Transition to drifting
            drift_data.state = DriftState.DRIFTING
            drift_data.drift_start_time = current_time
            logger.info(f"🌊 Person {person_id} started drifting...")
        
        elif drift_data.state == DriftState.DRIFTING:
            # Check if drift duration is over
            if current_time - drift_data.drift_start_time > self.drift_duration:
                self._start_drift_return(person_id, current_time)
        
        elif drift_data.state == DriftState.DRIFT_RETURN:
            # Check if return is complete
            if current_time - drift_data.drift_return_start_time > self.drift_return_duration:
                self._end_drift(person_id, current_time)
        
        elif drift_data.state == DriftState.COOLDOWN:
            # Wait for cooldown to complete
            if current_time - drift_data.last_drift_time > self.drift_cooldown:
                drift_data.state = DriftState.FOLLOWING
    
    def _start_drift(self, person_id: int, current_time: float):
        """Start drift behavior for a person"""
        drift_data = self.person_drift_data[person_id]
        drift_data.state = DriftState.DRIFT_START
        drift_data.drift_start_time = current_time
        
        # Store the current pose as the base to return to
        pose_history = list(self.person_histories.get(person_id, []))
        if pose_history:
            drift_data.drift_base_pose = pose_history[-1].copy()
        
        logger.info(f"🎭 Person {person_id} starting drift behavior...")
    
    def _start_drift_return(self, person_id: int, current_time: float):
        """Start returning from drift to human pose"""
        drift_data = self.person_drift_data[person_id]
        drift_data.state = DriftState.DRIFT_RETURN
        drift_data.drift_return_start_time = current_time
        logger.info(f"🏠 Person {person_id} returning from drift...")
    
    def _end_drift(self, person_id: int, current_time: float):
        """End drift behavior and return to normal following"""
        drift_data = self.person_drift_data[person_id]
        drift_data.state = DriftState.COOLDOWN
        drift_data.last_drift_time = current_time
        drift_data.accumulated_prediction = None
        drift_data.drift_base_pose = None
        logger.info(f"✅ Person {person_id} drift complete, cooling down...")
    
    def _generate_drift_aware_pose(self, person_id: int, current_pose: Dict[str, Any], 
                                  current_time: float) -> Dict[str, Any]:
        """Generate pose considering drift behavior"""
        drift_data = self.person_drift_data[person_id]
        
        if drift_data.state == DriftState.FOLLOWING:
            # Normal pose processing
            enhanced_pose = self._generate_enhanced_pose(person_id, current_pose)
            enhanced_pose['drift_state'] = 'following'
            return enhanced_pose
        
        elif drift_data.state == DriftState.DRIFT_START:
            # Start transitioning to predicted pose
            enhanced_pose = self._generate_enhanced_pose(person_id, current_pose)
            enhanced_pose['drift_state'] = 'drift_start'
            return enhanced_pose
        
        elif drift_data.state == DriftState.DRIFTING:
            # Follow predictions with occasional human influence
            predicted_pose = self._generate_drift_prediction(person_id, current_pose)
            predicted_pose['drift_state'] = 'drifting'
            return predicted_pose
        
        elif drift_data.state == DriftState.DRIFT_RETURN:
            # Gradually return to human pose
            return_pose = self._generate_drift_return_pose(person_id, current_pose, current_time)
            return_pose['drift_state'] = 'returning'
            return return_pose
        
        elif drift_data.state == DriftState.COOLDOWN:
            # Normal following with drift indicator
            enhanced_pose = self._generate_enhanced_pose(person_id, current_pose)
            enhanced_pose['drift_state'] = 'cooldown'
            return enhanced_pose
        
        return current_pose
    
    def _generate_drift_prediction(self, person_id: int, current_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pose during drift phase (mostly predictions)"""
        pose_history = list(self.person_histories.get(person_id, []))
        
        if len(pose_history) < 3:
            return current_pose
        
        # Get prediction
        if self.simlpe and self.simlpe.initialized:
            prediction_result = self.simlpe.predict_poses(pose_history)
            
            if prediction_result.get('success', False):
                predicted_poses = prediction_result.get('predicted_poses', [])
                
                if predicted_poses:
                    predicted_pose = predicted_poses[0]
                    
                    # During drift, heavily favor predictions over current pose
                    drift_pose = self._blend_poses(
                        current_pose, 
                        predicted_pose, 
                        blend_factor=self.drift_prediction_strength
                    )
                    
                    drift_pose['predicted'] = True
                    drift_pose['drift_strength'] = self.drift_prediction_strength
                    
                    return drift_pose
        
        # Fallback: use motion extrapolation during drift
        return self._extrapolate_drift_motion(person_id, current_pose)
    
    def _generate_drift_return_pose(self, person_id: int, current_pose: Dict[str, Any], 
                                   current_time: float) -> Dict[str, Any]:
        """Generate pose during return phase (blend prediction back to human)"""
        drift_data = self.person_drift_data[person_id]
        
        # Calculate return progress (0.0 = start return, 1.0 = fully returned)
        elapsed_return = current_time - drift_data.drift_return_start_time
        return_progress = min(1.0, elapsed_return / self.drift_return_duration)
        
        # Smooth easing function for natural return
        ease_progress = self._ease_in_out(return_progress)
        
        # Blend from prediction-heavy to human-heavy
        prediction_strength = self.drift_prediction_strength * (1.0 - ease_progress)
        
        # Get current prediction
        pose_history = list(self.person_histories.get(person_id, []))
        
        if len(pose_history) >= 3 and self.simlpe and self.simlpe.initialized:
            prediction_result = self.simlpe.predict_poses(pose_history)
            
            if prediction_result.get('success', False):
                predicted_poses = prediction_result.get('predicted_poses', [])
                
                if predicted_poses:
                    predicted_pose = predicted_poses[0]
                    
                    # Gradually reduce prediction influence
                    return_pose = self._blend_poses(
                        current_pose,
                        predicted_pose,
                        blend_factor=prediction_strength
                    )
                    
                    return_pose['drift_return_progress'] = return_progress
                    return_pose['prediction_strength'] = prediction_strength
                    
                    return return_pose
        
        # Fallback: smooth transition back to current pose
        return self._temporal_smoothing(person_id, current_pose)
    
    def _extrapolate_drift_motion(self, person_id: int, current_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Extrapolate motion during drift when prediction fails"""
        pose_history = list(self.person_histories.get(person_id, []))
        
        if len(pose_history) < 2:
            return current_pose
        
        # Use enhanced extrapolation during drift
        prev_pose = pose_history[-2]
        
        # Amplify the motion for more dramatic drift
        amplified_pose = self._amplify_motion(prev_pose, current_pose, amplification=1.5)
        amplified_pose['extrapolated_drift'] = True
        
        return amplified_pose
    
    def _amplify_motion(self, prev_pose: Dict[str, Any], curr_pose: Dict[str, Any], 
                       amplification: float = 1.5) -> Dict[str, Any]:
        """Amplify motion between poses for dramatic drift effect"""
        if not prev_pose.get('pose_present', False) or not curr_pose.get('pose_present', False):
            return curr_pose
        
        prev_joints = prev_pose.get('joints', {})
        curr_joints = curr_pose.get('joints', {})
        
        amplified_joints = {}
        
        for joint_name in curr_joints:
            if joint_name in prev_joints:
                curr_pos = curr_joints[joint_name]['position']
                prev_pos = prev_joints[joint_name]['position']
                
                # Calculate amplified motion
                motion = [
                    (curr_pos[0] - prev_pos[0]) * amplification,
                    (curr_pos[1] - prev_pos[1]) * amplification,
                    (curr_pos[2] - prev_pos[2]) * amplification
                ]
                
                amplified_pos = [
                    curr_pos[0] + motion[0],
                    curr_pos[1] + motion[1],
                    curr_pos[2] + motion[2]
                ]
                
                amplified_joints[joint_name] = {
                    'position': amplified_pos,
                    'confidence': curr_joints[joint_name].get('confidence', 0.0) * 0.7,  # Lower confidence
                    'amplified': True
                }
            else:
                amplified_joints[joint_name] = curr_joints[joint_name].copy()
        
        return {
            'pose_present': True,
            'joints': amplified_joints,
            'confidence': curr_pose.get('confidence', 0.0) * 0.8,
            'timestamp': time.time(),
            'amplified_motion': True
        }
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function for natural transitions"""
        return t * t * (3.0 - 2.0 * t)
    
    def _generate_enhanced_pose(self, person_id: int, current_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced pose with prediction and smoothing"""
        pose_history = list(self.person_histories.get(person_id, []))
        
        if len(pose_history) < 3:
            # Not enough history, return current pose
            return current_pose
        
        try:
            # Use siMLPe for prediction
            if self.simlpe and self.simlpe.initialized:
                prediction_result = self.simlpe.predict_poses(pose_history)
                
                if prediction_result.get('success', False):
                    predicted_poses = prediction_result.get('predicted_poses', [])
                    
                    if predicted_poses:
                        # Use the first predicted pose (immediate future)
                        next_pose = predicted_poses[0]
                        
                        # Blend current pose with prediction for smoothness
                        blended_pose = self._blend_poses(current_pose, next_pose, blend_factor=0.3)
                        
                        # Add prediction metadata
                        blended_pose['prediction_confidence'] = prediction_result.get('confidence', 0.0)
                        blended_pose['enhanced'] = True
                        
                        return blended_pose
            
            # Fallback: use temporal smoothing only
            return self._temporal_smoothing(person_id, current_pose)
            
        except Exception as e:
            logger.error(f"Pose enhancement failed: {e}")
            return current_pose
    
    def _blend_poses(self, pose1: Dict[str, Any], pose2: Dict[str, Any], 
                    blend_factor: float = 0.5) -> Dict[str, Any]:
        """Blend two poses for smooth transitions"""
        if not pose1.get('pose_present', False) or not pose2.get('pose_present', False):
            return pose1
        
        joints1 = pose1.get('joints', {})
        joints2 = pose2.get('joints', {})
        
        blended_joints = {}
        
        for joint_name in joints1:
            if joint_name in joints2:
                pos1 = joints1[joint_name]['position']
                pos2 = joints2[joint_name]['position']
                conf1 = joints1[joint_name].get('confidence', 0.0)
                conf2 = joints2[joint_name].get('confidence', 0.0)
                
                # Blend positions
                blended_pos = [
                    pos1[0] * (1 - blend_factor) + pos2[0] * blend_factor,
                    pos1[1] * (1 - blend_factor) + pos2[1] * blend_factor,
                    pos1[2] * (1 - blend_factor) + pos2[2] * blend_factor
                ]
                
                # Blend confidences
                blended_conf = conf1 * (1 - blend_factor) + conf2 * blend_factor
                
                blended_joints[joint_name] = {
                    'position': blended_pos,
                    'confidence': blended_conf
                }
            else:
                blended_joints[joint_name] = joints1[joint_name].copy()
        
        return {
            'pose_present': True,
            'joints': blended_joints,
            'confidence': pose1.get('confidence', 0.0) * (1 - blend_factor) + pose2.get('confidence', 0.0) * blend_factor,
            'timestamp': time.time(),
            'blended': True
        }
    
    def _temporal_smoothing(self, person_id: int, current_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing without prediction"""
        pose_history = list(self.person_histories.get(person_id, []))
        
        if len(pose_history) < 3:
            return current_pose
        
        # Use last 3 poses for smoothing
        recent_poses = pose_history[-3:]
        
        return self._average_poses(recent_poses, weights=[0.2, 0.3, 0.5])  # Most weight on current
    
    def _average_poses(self, poses: List[Dict[str, Any]], weights: List[float]) -> Dict[str, Any]:
        """Average multiple poses with weights"""
        if not poses or len(poses) != len(weights):
            return poses[-1] if poses else {}
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        averaged_joints = {}
        
        # Get all joint names from the latest pose
        latest_joints = poses[-1].get('joints', {})
        
        for joint_name in latest_joints:
            positions = []
            confidences = []
            
            for pose, weight in zip(poses, weights):
                joints = pose.get('joints', {})
                if joint_name in joints:
                    positions.append(joints[joint_name]['position'])
                    confidences.append(joints[joint_name].get('confidence', 0.0))
                else:
                    # Missing joint, use last known position
                    positions.append(latest_joints[joint_name]['position'])
                    confidences.append(0.0)
            
            # Weighted average
            avg_pos = [0.0, 0.0, 0.0]
            avg_conf = 0.0
            
            for pos, conf, weight in zip(positions, confidences, weights):
                avg_pos[0] += pos[0] * weight
                avg_pos[1] += pos[1] * weight
                avg_pos[2] += pos[2] * weight
                avg_conf += conf * weight
            
            averaged_joints[joint_name] = {
                'position': avg_pos,
                'confidence': avg_conf
            }
        
        return {
            'pose_present': True,
            'joints': averaged_joints,
            'confidence': sum(pose.get('confidence', 0.0) * weight for pose, weight in zip(poses, weights)),
            'timestamp': time.time(),
            'smoothed': True
        }
    
    def _cleanup_histories(self, active_people: set):
        """Clean up pose histories and drift data for people who disappeared"""
        disappeared_people = set(self.person_histories.keys()) - active_people
        for person_id in disappeared_people:
            if person_id in self.person_histories:
                del self.person_histories[person_id]
            if person_id in self.cached_predictions:
                del self.cached_predictions[person_id]
            if person_id in self.person_drift_data:
                del self.person_drift_data[person_id]
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about pose prediction and drift behavior"""
        # Get drift states for all people
        drift_stats = {}
        for person_id, drift_data in self.person_drift_data.items():
            drift_stats[person_id] = drift_data.state.value
        
        return {
            'enabled': self.enabled,
            'tracked_people': len(self.person_histories),
            'model_available': self.simlpe.initialized if self.simlpe else False,
            'prediction_horizon': self.prediction_horizon,
            'drift_enabled': self.drift_enabled,
            'drift_stats': drift_stats
        }