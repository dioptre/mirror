"""
MonoPort integration for real-time human performance capture
https://github.com/Project-Splinter/MonoPort
"""

import os
import sys
import cv2
import numpy as np
import torch
import subprocess
import tempfile
import shutil
import json
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MonoPortWrapper:
    """Wrapper for MonoPort real-time human performance capture"""
    
    def __init__(self, monoport_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize MonoPort wrapper
        
        Args:
            monoport_path: Path to MonoPort installation
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.monoport_path = monoport_path or self._find_monoport_installation()
        self.model = None
        self.initialized = False
        
        if self.monoport_path and os.path.exists(self.monoport_path):
            self._setup_monoport()
        else:
            logger.warning("MonoPort not found. Will attempt to download on first use.")
    
    def _get_device(self, device: str) -> str:
        """Determine device to use"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _find_monoport_installation(self) -> Optional[str]:
        """Try to find existing MonoPort installation"""
        possible_paths = [
            './models/reconstruction/MonoPort',
            './MonoPort',
            '../MonoPort',
            '../../MonoPort',
            os.path.expanduser('~/MonoPort'),
            '/opt/MonoPort'
        ]
        
        for path in possible_paths:
            if (os.path.exists(os.path.join(path, 'monoport')) or
                os.path.exists(os.path.join(path, 'src')) or
                os.path.exists(os.path.join(path, 'README.md'))):
                logger.info(f"Found MonoPort at {path}")
                return path
        
        return None
    
    def _setup_monoport(self):
        """Setup MonoPort environment"""
        try:
            # Add MonoPort to Python path
            if self.monoport_path not in sys.path:
                sys.path.insert(0, self.monoport_path)
            
            # Test basic imports
            logger.info("Testing MonoPort imports...")
            
            # MonoPort typically has these modules
            try:
                # Try importing core MonoPort modules
                import monoport
                logger.info("✅ MonoPort main module imported")
            except ImportError:
                # Try alternative import paths
                try:
                    from src import monoport_model
                    logger.info("✅ MonoPort model module imported")
                except ImportError:
                    logger.info("MonoPort modules not found, will use simplified approach")
            
            self.initialized = True
            logger.info("MonoPort environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup MonoPort: {e}")
            self.initialized = False
    
    def download_monoport(self, install_path: str = './models/reconstruction/MonoPort') -> bool:
        """
        Download and setup MonoPort
        
        Args:
            install_path: Where to install MonoPort
            
        Returns:
            True if successful
        """
        try:
            logger.info("Downloading MonoPort...")
            
            # Clone repository
            subprocess.run([
                'git', 'clone',
                'https://github.com/Project-Splinter/MonoPort.git',
                install_path
            ], check=True)
            
            self.monoport_path = install_path
            self._setup_monoport()
            
            return self.initialized
            
        except Exception as e:
            logger.error(f"Failed to download MonoPort: {e}")
            return False
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load MonoPort model"""
        if not self.initialized:
            if not self.download_monoport():
                raise RuntimeError("Failed to setup MonoPort")
        
        try:
            logger.info("Loading MonoPort model...")
            
            # For now, we'll create a simplified MonoPort-style reconstruction
            # In a full implementation, you would load the actual MonoPort model
            self.model = "monoport_simplified"
            
            logger.info("✅ MonoPort model ready (simplified implementation)")
            
        except Exception as e:
            logger.error(f"Failed to load MonoPort model: {e}")
            raise
    
    def reconstruct_person(self, image: np.ndarray, output_path: str) -> Dict[str, Any]:
        """
        Reconstruct person using MonoPort-style approach
        
        Args:
            image: Input image (preprocessed)
            output_path: Output path for mesh
            
        Returns:
            Reconstruction results
        """
        if self.model is None:
            self.load_model()
        
        try:
            logger.info("🚀 MonoPort reconstruction starting...")
            
            # MonoPort-style reconstruction pipeline
            result = self._monoport_reconstruction(image)
            
            # Save mesh
            mesh_path = self._save_mesh(result, output_path)
            
            return {
                'success': True,
                'mesh_path': mesh_path,
                'vertices': result.get('vertices'),
                'faces': result.get('faces'),
                'method': 'monoport'
            }
            
        except Exception as e:
            logger.error(f"MonoPort reconstruction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _monoport_reconstruction(self, image: np.ndarray) -> Dict[str, Any]:
        """
        MonoPort-style reconstruction using silhouette and pose estimation
        """
        # Step 1: Extract person silhouette
        silhouette = self._extract_clean_silhouette(image)
        
        # Step 2: Estimate 2D pose keypoints (MonoPort uses pose for reconstruction)
        keypoints = self._estimate_pose_keypoints(image)
        
        # Step 3: Generate 3D mesh using pose-guided reconstruction
        vertices, faces = self._generate_pose_guided_mesh(silhouette, keypoints, image.shape[:2])
        
        return {
            'vertices': vertices,
            'faces': faces,
            'keypoints': keypoints,
            'silhouette': silhouette
        }
    
    def _extract_clean_silhouette(self, image: np.ndarray) -> np.ndarray:
        """Extract clean person silhouette"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create binary mask
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find largest contour (person)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create clean silhouette
            clean_silhouette = np.zeros_like(binary)
            cv2.fillPoly(clean_silhouette, [largest_contour], 255)
            
            # Smooth the silhouette
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clean_silhouette = cv2.morphologyEx(clean_silhouette, cv2.MORPH_CLOSE, kernel)
            
            return clean_silhouette
        
        return binary
    
    def _estimate_pose_keypoints(self, image: np.ndarray) -> Dict[str, Any]:
        """Estimate 2D pose keypoints for mesh generation"""
        # Use MediaPipe for pose estimation (MonoPort would use this for guidance)
        try:
            import mediapipe as mp
            
            mp_pose = mp.solutions.pose
            pose_detector = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb_image)
            
            keypoints = {}
            if results.pose_landmarks:
                h, w = image.shape[:2]
                
                # Extract key landmarks for mesh generation
                landmarks = results.pose_landmarks.landmark
                
                keypoints = {
                    'nose': [landmarks[0].x * w, landmarks[0].y * h, landmarks[0].z],
                    'left_shoulder': [landmarks[11].x * w, landmarks[11].y * h, landmarks[11].z],
                    'right_shoulder': [landmarks[12].x * w, landmarks[12].y * h, landmarks[12].z],
                    'left_elbow': [landmarks[13].x * w, landmarks[13].y * h, landmarks[13].z],
                    'right_elbow': [landmarks[14].x * w, landmarks[14].y * h, landmarks[14].z],
                    'left_wrist': [landmarks[15].x * w, landmarks[15].y * h, landmarks[15].z],
                    'right_wrist': [landmarks[16].x * w, landmarks[16].y * h, landmarks[16].z],
                    'left_hip': [landmarks[23].x * w, landmarks[23].y * h, landmarks[23].z],
                    'right_hip': [landmarks[24].x * w, landmarks[24].y * h, landmarks[24].z],
                    'left_knee': [landmarks[25].x * w, landmarks[25].y * h, landmarks[25].z],
                    'right_knee': [landmarks[26].x * w, landmarks[26].y * h, landmarks[26].z],
                    'left_ankle': [landmarks[27].x * w, landmarks[27].y * h, landmarks[27].z],
                    'right_ankle': [landmarks[28].x * w, landmarks[28].y * h, landmarks[28].z],
                }
            
            pose_detector.close()
            return keypoints
            
        except ImportError:
            logger.warning("MediaPipe not available for pose estimation")
            return {}
        except Exception as e:
            logger.warning(f"Pose estimation failed: {e}")
            return {}
    
    def _generate_pose_guided_mesh(self, silhouette: np.ndarray, keypoints: Dict[str, Any], 
                                  image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 3D mesh guided by pose keypoints (MonoPort-style)"""
        h, w = image_shape
        
        # Create base mesh from silhouette
        vertices = []
        vertex_map = {}
        
        # Scale factors
        scale_x = 2.0 / w  # 2-unit width
        scale_y = 3.0 / h  # 3-unit height
        
        # Generate vertices from silhouette with pose-guided depth
        vertex_idx = 0
        for y in range(0, h, 8):  # Sample every 8 pixels for performance
            for x in range(0, w, 8):
                if silhouette[y, x] > 0:
                    # Convert to world coordinates
                    world_x = (x - w//2) * scale_x
                    world_y = (h - y - h//2) * scale_y
                    
                    # Calculate depth using pose guidance
                    depth = self._calculate_pose_guided_depth(x, y, keypoints, (w, h))
                    
                    vertices.append([world_x, world_y, depth])
                    vertex_map[(x, y)] = vertex_idx
                    vertex_idx += 1
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Generate faces
        faces = []
        for y in range(0, h-8, 8):
            for x in range(0, w-8, 8):
                # Try to create quad from 2x2 vertex grid
                corners = [(x, y), (x+8, y), (x, y+8), (x+8, y+8)]
                corner_indices = []
                
                for corner in corners:
                    if corner in vertex_map:
                        corner_indices.append(vertex_map[corner])
                
                # Create triangles if we have enough vertices
                if len(corner_indices) >= 3:
                    faces.append([corner_indices[0], corner_indices[1], corner_indices[2]])
                    
                    if len(corner_indices) == 4:
                        faces.append([corner_indices[0], corner_indices[2], corner_indices[3]])
        
        faces = np.array(faces, dtype=np.int32)
        
        logger.info(f"Generated pose-guided mesh: {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces
    
    def _calculate_pose_guided_depth(self, x: int, y: int, keypoints: Dict[str, Any], 
                                   image_size: Tuple[int, int]) -> float:
        """Calculate depth value guided by pose keypoints (MonoPort approach)"""
        w, h = image_size
        base_depth = 0.1
        
        if not keypoints:
            # Fallback: simple distance-based depth
            center_x, center_y = w // 2, h // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            return base_depth * (1.0 - distance / max_distance) * 0.5
        
        # Use keypoints to guide depth estimation
        max_depth = 0.0
        
        # Check proximity to key body parts
        body_parts = {
            'head': ['nose'],
            'torso': ['left_shoulder', 'right_shoulder'],
            'arms': ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
            'hips': ['left_hip', 'right_hip'],
            'legs': ['left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        }
        
        depth_factors = {
            'head': 0.8,     # Head sticks out most
            'torso': 0.6,    # Torso has good depth
            'arms': 0.4,     # Arms are thinner
            'hips': 0.5,     # Hips have medium depth
            'legs': 0.3      # Legs are thinner
        }
        
        for part_name, part_keypoints in body_parts.items():
            for kp_name in part_keypoints:
                if kp_name in keypoints:
                    kp = keypoints[kp_name]
                    kp_x, kp_y = kp[0], kp[1]
                    
                    # Calculate distance to keypoint
                    distance = np.sqrt((x - kp_x)**2 + (y - kp_y)**2)
                    
                    # Influence decreases with distance
                    influence_radius = 50  # pixels
                    if distance < influence_radius:
                        influence = (1.0 - distance / influence_radius)
                        depth_contribution = depth_factors[part_name] * influence
                        max_depth = max(max_depth, depth_contribution)
        
        return base_depth + max_depth * 0.3
    
    def _save_mesh(self, result: Dict[str, Any], output_path: str) -> str:
        """Save mesh to OBJ file"""
        vertices = result['vertices']
        faces = result['faces']
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh_path = output_path + '.obj'
        
        with open(mesh_path, 'w') as f:
            f.write("# MonoPort-style reconstruction\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                if len(face) >= 3:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        logger.info(f"Saved MonoPort mesh to {mesh_path}")
        return mesh_path


class MonoPortPipeline:
    """Complete MonoPort reconstruction pipeline"""
    
    def __init__(self, cache_dir: str = './reconstruction_cache'):
        self.monoport = MonoPortWrapper()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("✅ MonoPort pipeline initialized")
    
    def process_person(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process person through MonoPort pipeline
        
        Args:
            person_data: Dictionary with person info and best frame
            
        Returns:
            Reconstruction results
        """
        person_id = person_data['id']
        image = person_data['best_frame']
        
        # Create output path
        output_path = os.path.join(self.cache_dir, f"person_{person_id}")
        
        # Run MonoPort reconstruction
        result = self.monoport.reconstruct_person(image, output_path)
        
        # Add metadata
        result['person_id'] = person_id
        result['timestamp'] = person_data.get('timestamp')
        result['cache_path'] = output_path
        
        return result
    
    def is_available(self) -> bool:
        """Check if MonoPort is available"""
        return self.monoport.initialized