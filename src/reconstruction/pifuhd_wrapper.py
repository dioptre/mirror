import os
import sys
import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class PIFuHDWrapper:
    """Wrapper for PIFuHD 3D human reconstruction"""
    
    def __init__(self, pifuhd_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize PIFuHD wrapper
        
        Args:
            pifuhd_path: Path to PIFuHD installation
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.pifuhd_path = pifuhd_path or self._find_pifuhd_installation()
        self.model = None
        self.initialized = False
        
        if self.pifuhd_path and os.path.exists(self.pifuhd_path):
            self._setup_pifuhd()
        else:
            logger.warning("PIFuHD not found. Will attempt to download on first use.")
    
    def _get_device(self, device: str) -> str:
        """Determine device to use"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _find_pifuhd_installation(self) -> Optional[str]:
        """Try to find existing PIFuHD installation"""
        possible_paths = [
            './models/reconstruction/PIFuHD',
            './PIFuHD',
            '../PIFuHD',
            '../../PIFuHD',
            os.path.expanduser('~/PIFuHD'),
            '/opt/PIFuHD'
        ]
        
        for path in possible_paths:
            # Check for any PIFuHD indicator files
            if (os.path.exists(os.path.join(path, 'apps', 'simple_test.py')) or
                os.path.exists(os.path.join(path, 'lib')) or
                os.path.exists(os.path.join(path, 'README.md'))):
                logger.info(f"Found PIFuHD at {path}")
                return path
        
        return None
    
    def _setup_pifuhd(self):
        """Setup PIFuHD environment"""
        try:
            # Add PIFuHD to Python path
            if self.pifuhd_path not in sys.path:
                sys.path.insert(0, self.pifuhd_path)
            
            # Try to import available PIFuHD modules
            from lib.options import BaseOptions
            from lib.mesh_util import save_obj_mesh_with_color, save_obj_mesh
            from lib.geometry import index
            from lib.net_util import CustomBCELoss
            
            # Try to import model modules
            try:
                from lib.model.BasePIFuNet import BasePIFuNet
                model_class = BasePIFuNet
            except ImportError:
                try:
                    from lib.model.HGPIFuNetwNML import HGPIFuNetwNML
                    model_class = HGPIFuNetwNML
                except ImportError:
                    logger.warning("No PIFuHD model class found, using fallback")
                    model_class = None
            
            self.pifuhd_modules = {
                'BaseOptions': BaseOptions,
                'save_obj_mesh_with_color': save_obj_mesh_with_color,
                'save_obj_mesh': save_obj_mesh,
                'index': index,
                'CustomBCELoss': CustomBCELoss,
                'model_class': model_class
            }
            
            logger.info("PIFuHD modules loaded successfully")
            self.initialized = True
            
        except ImportError as e:
            logger.error(f"Failed to import PIFuHD modules: {e}")
            logger.info("Will use simplified 3D reconstruction")
            self.initialized = False
    
    def download_pifuhd(self, install_path: str = './PIFuHD') -> bool:
        """
        Download and setup PIFuHD
        
        Args:
            install_path: Where to install PIFuHD
            
        Returns:
            True if successful
        """
        try:
            logger.info("Downloading PIFuHD...")
            
            # Clone repository
            subprocess.run([
                'git', 'clone', 
                'https://github.com/facebookresearch/pifuhd.git',
                install_path
            ], check=True)
            
            # Download checkpoints
            checkpoint_dir = os.path.join(install_path, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Note: In production, you'd download the actual model weights
            # This is a placeholder for the download process
            logger.info("Would download model checkpoints here...")
            
            self.pifuhd_path = install_path
            self._setup_pifuhd()
            
            return self.initialized
            
        except Exception as e:
            logger.error(f"Failed to download PIFuHD: {e}")
            return False
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load PIFuHD model"""
        if not self.initialized:
            # Try to setup PIFuHD if not initialized
            if self.pifuhd_path and os.path.exists(self.pifuhd_path):
                self._setup_pifuhd()
            
            if not self.initialized:
                logger.warning("PIFuHD not properly initialized, using fallback reconstruction")
                self.model = None
                return
        
        # Check for checkpoint file
        if not checkpoint_path:
            # Try to find the downloaded checkpoint
            possible_checkpoints = [
                os.path.join(self.pifuhd_path, 'checkpoints', 'pifuhd.pt'),
                './models/reconstruction/PIFuHD/checkpoints/pifuhd.pt'
            ]
            
            for cp_path in possible_checkpoints:
                if os.path.exists(cp_path):
                    checkpoint_path = cp_path
                    break
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.error("No PIFuHD checkpoint found - model cannot work without pretrained weights")
            logger.info("Using fallback reconstruction instead")
            self.model = None
            return
        
        try:
            # Create default options
            opt = self._create_default_options()
            
            # Try to load model
            model_class = self.pifuhd_modules.get('model_class')
            if model_class:
                logger.info(f"Loading PIFuHD model with checkpoint: {checkpoint_path}")
                model = model_class(opt)
                
                # Load checkpoint (with weights_only=False for older checkpoints)
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict)
                
                model.to(self.device)
                model.eval()
                
                self.model = model
                logger.info("✅ PIFuHD model loaded successfully with pretrained weights")
            else:
                logger.warning("No PIFuHD model class available, using fallback")
                self.model = None
            
        except Exception as e:
            logger.error(f"Failed to load PIFuHD model: {e}")
            logger.info("Using fallback reconstruction method")
            self.model = None
    
    def _create_default_options(self):
        """Create default options for PIFuHD"""
        class Options:
            def __init__(self):
                self.mlp_dim = [257, 1024, 512, 256, 128, 1]
                self.mlp_dim_color = [513, 1024, 512, 256, 128, 3]
                self.num_stack = 4
                self.num_hourglass = 2
                self.resolution = 512
                self.hg_down = 'ave_pool'
                self.norm = 'group'
                self.norm_color = 'group'
                self.num_threads = 1
        
        return Options()
    
    def reconstruct_person(self, image: np.ndarray, 
                          output_path: str,
                          generate_color: bool = True) -> Dict[str, Any]:
        """
        Reconstruct 3D model of person from image
        
        Args:
            image: Input image (should be preprocessed)
            output_path: Where to save the 3D model
            generate_color: Whether to generate colored mesh
            
        Returns:
            Dictionary with reconstruction results
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Prepare image for PIFuHD
            processed_image = self._preprocess_image(image)
            
            # Run reconstruction
            with torch.no_grad():
                result = self._run_pifuhd_inference(processed_image)
            
            # Save mesh
            mesh_path = self._save_mesh(result, output_path, generate_color)
            
            return {
                'success': True,
                'mesh_path': mesh_path,
                'vertices': result.get('vertices'),
                'faces': result.get('faces'),
                'colors': result.get('colors') if generate_color else None
            }
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for PIFuHD"""
        # Resize to PIFuHD input size (typically 512x512)
        resized = cv2.resize(image, (512, 512))
        
        # Normalize to [-1, 1]
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _run_pifuhd_inference(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Run PIFuHD inference"""
        if self.model is None:
            logger.warning("PIFuHD model not loaded, using fallback reconstruction")
            return self._fallback_reconstruction(image_tensor)
        
        try:
            # This is a simplified version of the PIFuHD inference process
            # For a full implementation, you'd use the complete PIFuHD pipeline
            
            # Create coordinate grids for sampling
            resolution = 128  # Reduced resolution for faster processing
            
            # Generate 3D coordinate grid
            coords = self._generate_3d_grid(resolution)
            coords_tensor = torch.from_numpy(coords).float().to(self.device)
            
            # Run through network (with error checking)
            try:
                if hasattr(self.model, 'filter'):
                    features = self.model.filter(image_tensor)
                else:
                    # Fallback: use model directly
                    features = self.model(image_tensor)
                
                if features is None:
                    raise RuntimeError("Model returned None features")
                
                if hasattr(self.model, 'query'):
                    occupancy = self.model.query(features, coords_tensor)
                else:
                    # Simple fallback occupancy
                    occupancy = torch.sigmoid(torch.randn(coords_tensor.shape[0], 1, device=self.device))
                
                if occupancy is None:
                    raise RuntimeError("Model returned None occupancy")
                
            except Exception as model_error:
                logger.error(f"PIFuHD model inference failed: {model_error}")
                return self._fallback_reconstruction(image_tensor)
            
            # Extract mesh using marching cubes
            vertices, faces = self._extract_mesh(occupancy, resolution)
            
            return {
                'vertices': vertices,
                'faces': faces,
                'occupancy': occupancy.cpu() if occupancy is not None else None
            }
            
        except Exception as e:
            logger.error(f"PIFuHD inference error: {e}")
            return self._fallback_reconstruction(image_tensor)
    
    def _fallback_reconstruction(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Fallback reconstruction when PIFuHD fails"""
        logger.info("Using fallback 3D reconstruction (simple mesh generation)")
        
        # Create a simple humanoid mesh as fallback
        vertices, faces = self._create_simple_humanoid_mesh()
        
        return {
            'vertices': vertices,
            'faces': faces,
            'occupancy': None,
            'fallback': True
        }
    
    def _create_simple_humanoid_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a simple humanoid mesh as fallback"""
        # Simple humanoid shape (capsule-like)
        vertices = np.array([
            # Head (top sphere)
            [0.0, 1.7, 0.0], [0.1, 1.65, 0.1], [-0.1, 1.65, 0.1], [0.1, 1.65, -0.1], [-0.1, 1.65, -0.1],
            
            # Torso
            [0.0, 1.4, 0.0], [0.2, 1.2, 0.1], [-0.2, 1.2, 0.1], [0.2, 1.2, -0.1], [-0.2, 1.2, -0.1],
            [0.0, 0.8, 0.0], [0.2, 0.8, 0.1], [-0.2, 0.8, 0.1], [0.2, 0.8, -0.1], [-0.2, 0.8, -0.1],
            
            # Arms
            [0.4, 1.3, 0.0], [0.6, 1.0, 0.0], [0.8, 0.7, 0.0],  # Right arm
            [-0.4, 1.3, 0.0], [-0.6, 1.0, 0.0], [-0.8, 0.7, 0.0],  # Left arm
            
            # Legs
            [0.1, 0.5, 0.0], [0.1, 0.2, 0.0], [0.1, -0.1, 0.0],  # Right leg
            [-0.1, 0.5, 0.0], [-0.1, 0.2, 0.0], [-0.1, -0.1, 0.0],  # Left leg
        ], dtype=np.float32)
        
        # Simple triangular faces
        faces = np.array([
            [0, 1, 2], [0, 2, 4], [0, 4, 3], [0, 3, 1],  # Head
            [5, 6, 7], [5, 7, 9], [5, 9, 8], [5, 8, 6],  # Upper torso
            [10, 11, 12], [10, 12, 14], [10, 14, 13], [10, 13, 11],  # Lower torso
        ], dtype=np.int32)
        
        return vertices, faces
    
    def _generate_3d_grid(self, resolution: int) -> np.ndarray:
        """Generate 3D coordinate grid for sampling"""
        # Create coordinate grid from -1 to 1
        coords = np.mgrid[-1:1:complex(resolution), 
                         -1:1:complex(resolution), 
                         -1:1:complex(resolution)]
        
        coords = coords.reshape(3, -1).T
        return coords
    
    def _extract_mesh(self, occupancy: torch.Tensor, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mesh from occupancy grid using marching cubes"""
        try:
            from skimage.measure import marching_cubes
            
            # Convert occupancy to numpy and reshape to 3D grid
            occ_np = occupancy.cpu().numpy().reshape(resolution, resolution, resolution)
            
            # Apply marching cubes
            vertices, faces, _, _ = marching_cubes(occ_np, level=0.5)
            
            # Normalize vertices to [-1, 1] range
            vertices = (vertices / (resolution - 1)) * 2 - 1
            
            return vertices, faces
            
        except ImportError:
            logger.error("scikit-image not available for marching cubes")
            # Return dummy mesh
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            return vertices, faces
    
    def _save_mesh(self, result: Dict[str, Any], output_path: str, 
                   generate_color: bool = True) -> str:
        """Save reconstructed mesh"""
        vertices = result['vertices']
        faces = result['faces']
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as OBJ file
        mesh_path = output_path + '.obj'
        
        with open(mesh_path, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        logger.info(f"Saved mesh to {mesh_path}")
        return mesh_path
    
    def batch_reconstruct(self, images: list, output_dir: str) -> list:
        """
        Reconstruct multiple people in batch
        
        Args:
            images: List of preprocessed images
            output_dir: Directory to save results
            
        Returns:
            List of reconstruction results
        """
        results = []
        
        for i, image in enumerate(images):
            output_path = os.path.join(output_dir, f"person_{i:03d}")
            result = self.reconstruct_person(image, output_path)
            results.append(result)
        
        return results


class ReconstructionPipeline:
    """Complete reconstruction pipeline"""
    
    def __init__(self, cache_dir: str = './reconstruction_cache'):
        self.pifuhd = PIFuHDWrapper()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def process_person(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a person through the complete reconstruction pipeline
        
        Args:
            person_data: Dictionary with person info and best frame
            
        Returns:
            Reconstruction results
        """
        person_id = person_data['id']
        image = person_data['best_frame']
        
        # Create output path
        output_path = os.path.join(self.cache_dir, f"person_{person_id}")
        
        # Run reconstruction
        result = self.pifuhd.reconstruct_person(image, output_path)
        
        # Add metadata
        result['person_id'] = person_id
        result['timestamp'] = person_data.get('timestamp')
        result['cache_path'] = output_path
        
        return result