"""
Direct PIFuHD integration using the actual PIFuHD recon wrapper
"""

import os
import sys
import tempfile
import shutil
import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PIFuHDDirect:
    """Direct integration with PIFuHD using their reconstruction pipeline"""
    
    def __init__(self, pifuhd_path: str = './models/reconstruction/PIFuHD'):
        self.pifuhd_path = Path(pifuhd_path)
        self.checkpoint_path = self.pifuhd_path / 'checkpoints' / 'pifuhd.pt'
        self.initialized = False
        
        if self.pifuhd_path.exists():
            self._setup_pifuhd()
    
    def _setup_pifuhd(self):
        """Setup PIFuHD environment"""
        try:
            # Add PIFuHD to Python path
            pifuhd_str = str(self.pifuhd_path)
            if pifuhd_str not in sys.path:
                sys.path.insert(0, pifuhd_str)
            
            # Test import
            from apps.recon import reconWrapper
            
            self.recon_wrapper = reconWrapper
            self.initialized = True
            logger.info("✅ PIFuHD direct wrapper initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import PIFuHD: {e}")
            self.initialized = False
    
    def reconstruct_from_image(self, image: np.ndarray, output_dir: str) -> Dict[str, Any]:
        """
        Reconstruct 3D model from image using PIFuHD
        
        Args:
            image: Input image
            output_dir: Output directory for results
            
        Returns:
            Reconstruction result
        """
        if not self.initialized:
            return {'success': False, 'error': 'PIFuHD not initialized'}
        
        if not self.checkpoint_path.exists():
            return {'success': False, 'error': 'PIFuHD checkpoint not found'}
        
        try:
            # Create temporary directories
            temp_input_dir = tempfile.mkdtemp(prefix='pifuhd_input_')
            temp_output_dir = tempfile.mkdtemp(prefix='pifuhd_output_')
            
            try:
                # Save input image in PIFuHD expected format
                input_image_path = os.path.join(temp_input_dir, 'input.png')
                
                # PIFuHD expects specific image format - resize to 512x512
                resized_image = cv2.resize(image, (512, 512))
                cv2.imwrite(input_image_path, resized_image)
                
                # Create keypoints file (required by PIFuHD)
                keypoints_path = os.path.join(temp_input_dir, 'input_keypoints.json')
                keypoints_data = self._generate_dummy_keypoints()
                
                import json
                with open(keypoints_path, 'w') as f:
                    json.dump(keypoints_data, f)
                
                logger.info(f"Created input files: {input_image_path}, {keypoints_path}")
                
                # Prepare PIFuHD command arguments
                cmd = [
                    '--dataroot', temp_input_dir,
                    '--results_path', temp_output_dir,
                    '--loadSize', '1024',
                    '--resolution', '512',
                    '--load_netMR_checkpoint_path', str(self.checkpoint_path),
                    '--start_id', '-1',
                    '--end_id', '-1'
                ]
                
                logger.info("🎯 Running PIFuHD reconstruction...")
                
                # Run PIFuHD reconstruction
                self.recon_wrapper(cmd, use_rect=False)
                
                # Find output files
                output_files = list(Path(temp_output_dir).glob('**/*.obj'))
                
                if output_files:
                    # Copy result to final output directory
                    os.makedirs(output_dir, exist_ok=True)
                    final_mesh_path = os.path.join(output_dir, 'mesh.obj')
                    shutil.copy2(output_files[0], final_mesh_path)
                    
                    logger.info(f"✅ PIFuHD reconstruction successful: {final_mesh_path}")
                    
                    return {
                        'success': True,
                        'mesh_path': final_mesh_path,
                        'method': 'pifuhd_direct'
                    }
                else:
                    logger.error("No output mesh found from PIFuHD")
                    return {'success': False, 'error': 'No output mesh generated'}
                
            finally:
                # Cleanup temporary directories
                shutil.rmtree(temp_input_dir, ignore_errors=True)
                shutil.rmtree(temp_output_dir, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"PIFuHD reconstruction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_dummy_keypoints(self) -> Dict[str, Any]:
        """Generate dummy OpenPose keypoints for PIFuHD"""
        # PIFuHD needs OpenPose format keypoints
        # Create a basic human pose keypoints (25 points in COCO format)
        keypoints_2d = [
            # Basic standing pose keypoints (x, y, confidence)
            256, 100, 0.9,  # nose
            256, 120, 0.9,  # neck  
            230, 150, 0.8,  # right shoulder
            200, 180, 0.7,  # right elbow
            180, 210, 0.6,  # right wrist
            282, 150, 0.8,  # left shoulder
            312, 180, 0.7,  # left elbow
            332, 210, 0.6,  # left wrist
            240, 200, 0.8,  # right hip
            235, 280, 0.7,  # right knee
            230, 360, 0.6,  # right ankle
            272, 200, 0.8,  # left hip
            277, 280, 0.7,  # left knee
            282, 360, 0.6,  # left ankle
            250, 90, 0.9,   # right eye
            262, 90, 0.9,   # left eye
            245, 95, 0.8,   # right ear
            267, 95, 0.8,   # left ear
            # Additional points to reach 25 (COCO format)
            225, 380, 0.5,  # big toe right
            287, 380, 0.5,  # big toe left
            220, 385, 0.5,  # small toe right
            292, 385, 0.5,  # small toe left
            228, 375, 0.5,  # heel right
            284, 375, 0.5,  # heel left
            240, 220, 0.6,  # extra point
        ]
        
        return {
            "version": 1.3,
            "people": [
                {
                    "person_id": [-1],
                    "pose_keypoints_2d": keypoints_2d,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                }
            ]
        }
    
    def is_available(self) -> bool:
        """Check if PIFuHD is available and ready"""
        return self.initialized and self.checkpoint_path.exists()


class PIFuHDDirectPipeline:
    """Pipeline using direct PIFuHD integration"""
    
    def __init__(self, cache_dir: str = './reconstruction_cache'):
        self.pifuhd = PIFuHDDirect()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def process_person(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process person using direct PIFuHD"""
        person_id = person_data['id']
        image = person_data['best_frame']
        
        if not self.pifuhd.is_available():
            logger.warning("PIFuHD not available, using fallback")
            return self._fallback_reconstruction(person_data)
        
        # Create output path
        output_path = os.path.join(self.cache_dir, f"person_{person_id}")
        
        # Run PIFuHD reconstruction
        result = self.pifuhd.reconstruct_from_image(image, output_path)
        
        # Add metadata
        result['person_id'] = person_id
        result['timestamp'] = person_data.get('timestamp')
        result['cache_path'] = output_path
        
        return result
    
    def _fallback_reconstruction(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback reconstruction when PIFuHD isn't available"""
        person_id = person_data['id']
        
        # Create a simple mesh file
        output_path = os.path.join(self.cache_dir, f"person_{person_id}")
        os.makedirs(output_path, exist_ok=True)
        
        mesh_path = os.path.join(output_path, 'mesh.obj')
        
        # Create simple humanoid OBJ file
        with open(mesh_path, 'w') as f:
            # Simple humanoid vertices
            f.write("# Simple humanoid mesh\n")
            f.write("v 0.0 1.7 0.0\n")      # Head
            f.write("v 0.0 1.4 0.0\n")      # Neck
            f.write("v 0.0 1.0 0.0\n")      # Torso top
            f.write("v 0.0 0.5 0.0\n")      # Torso bottom
            f.write("v 0.0 0.0 0.0\n")      # Base
            f.write("v 0.3 1.2 0.0\n")      # Right shoulder
            f.write("v 0.5 0.8 0.0\n")      # Right elbow
            f.write("v 0.7 0.4 0.0\n")      # Right hand
            f.write("v -0.3 1.2 0.0\n")     # Left shoulder
            f.write("v -0.5 0.8 0.0\n")     # Left elbow
            f.write("v -0.7 0.4 0.0\n")     # Left hand
            f.write("v 0.1 0.2 0.0\n")      # Right hip
            f.write("v 0.1 -0.2 0.0\n")     # Right knee
            f.write("v 0.1 -0.6 0.0\n")     # Right foot
            f.write("v -0.1 0.2 0.0\n")     # Left hip
            f.write("v -0.1 -0.2 0.0\n")    # Left knee
            f.write("v -0.1 -0.6 0.0\n")    # Left foot
            
            # Simple faces
            f.write("f 1 2 3\n")
            f.write("f 2 3 4\n")
            f.write("f 3 4 5\n")
        
        return {
            'success': True,
            'mesh_path': mesh_path,
            'person_id': person_id,
            'method': 'fallback',
            'timestamp': person_data.get('timestamp'),
            'cache_path': output_path
        }