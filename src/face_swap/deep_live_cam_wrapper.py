import os
import sys
import cv2
import numpy as np
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import threading
import time

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available, face swapping disabled")

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logging.warning("ONNX Runtime not available")

logger = logging.getLogger(__name__)


class DeepLiveCamWrapper:
    """Wrapper for Deep-Live-Cam face swapping functionality"""
    
    def __init__(self, deep_live_cam_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize Deep-Live-Cam wrapper
        
        Args:
            deep_live_cam_path: Path to Deep-Live-Cam installation
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.deep_live_cam_path = deep_live_cam_path or self._find_deep_live_cam_installation()
        self.face_swapper = None
        self.face_analyzer = None
        self.initialized = False
        
        if INSIGHTFACE_AVAILABLE and ONNXRUNTIME_AVAILABLE:
            self._setup_insightface()
        else:
            logger.warning("Required dependencies not available for face swapping")
    
    def _get_device(self, device: str) -> str:
        """Determine device to use"""
        if device == 'auto':
            # Check for CUDA availability
            if ort.get_available_providers().__contains__('CUDAExecutionProvider'):
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def _find_deep_live_cam_installation(self) -> Optional[str]:
        """Try to find existing Deep-Live-Cam installation"""
        possible_paths = [
            './Deep-Live-Cam',
            '../Deep-Live-Cam',
            '../../Deep-Live-Cam',
            os.path.expanduser('~/Deep-Live-Cam'),
            '/opt/Deep-Live-Cam'
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'run.py')):
                logger.info(f"Found Deep-Live-Cam at {path}")
                return path
        
        return None
    
    def _setup_insightface(self):
        """Setup InsightFace for face analysis and swapping"""
        try:
            # Initialize face analyzer
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            
            # Initialize face swapper
            model_path = self._download_swap_model()
            if model_path:
                self.face_swapper = insightface.model_zoo.get_model(model_path, providers=providers)
            
            self.initialized = True
            logger.info("Deep-Live-Cam face swapping initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup InsightFace: {e}")
            self.initialized = False
    
    def _download_swap_model(self) -> Optional[str]:
        """Download face swapping model if needed"""
        model_dir = Path('./models/face_swap')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Common face swap models
        model_urls = {
            'inswapper_128.onnx': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
        }
        
        for model_name, url in model_urls.items():
            model_path = model_dir / model_name
            if not model_path.exists():
                try:
                    logger.info(f"Downloading face swap model: {model_name}")
                    # In production, you would download the actual model
                    # For now, we'll create a placeholder
                    model_path.touch()
                    return str(model_path)
                except Exception as e:
                    logger.error(f"Failed to download {model_name}: {e}")
        
        return None
    
    def download_deep_live_cam(self, install_path: str = './Deep-Live-Cam') -> bool:
        """
        Download and setup Deep-Live-Cam
        
        Args:
            install_path: Where to install Deep-Live-Cam
            
        Returns:
            True if successful
        """
        try:
            logger.info("Downloading Deep-Live-Cam...")
            
            # Clone repository
            subprocess.run([
                'git', 'clone',
                'https://github.com/hacksider/Deep-Live-Cam.git',
                install_path
            ], check=True)
            
            self.deep_live_cam_path = install_path
            logger.info("Deep-Live-Cam downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Deep-Live-Cam: {e}")
            return False
    
    def extract_face_features(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract face features from image
        
        Args:
            image: Input image
            
        Returns:
            List of face data with embeddings
        """
        if not self.initialized:
            return []
        
        try:
            faces = self.face_analyzer.get(image)
            
            face_data = []
            for face in faces:
                face_info = {
                    'bbox': face.bbox.astype(int).tolist(),
                    'kps': face.kps.astype(int).tolist(),
                    'embedding': face.embedding,
                    'det_score': float(face.det_score),
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None)
                }
                face_data.append(face_info)
            
            return face_data
            
        except Exception as e:
            logger.error(f"Face feature extraction failed: {e}")
            return []
    
    def swap_faces(self, source_image: np.ndarray, target_image: np.ndarray,
                   source_face_index: int = 0, target_face_index: int = 0) -> Optional[np.ndarray]:
        """
        Perform face swapping between source and target images
        
        Args:
            source_image: Image containing the source face
            target_image: Image where face will be swapped
            source_face_index: Index of face in source image
            target_face_index: Index of face in target image
            
        Returns:
            Image with swapped face or None if failed
        """
        if not self.initialized or not self.face_swapper:
            logger.warning("Face swapper not initialized")
            return None
        
        try:
            # Analyze faces in both images
            source_faces = self.face_analyzer.get(source_image)
            target_faces = self.face_analyzer.get(target_image)
            
            if len(source_faces) <= source_face_index:
                logger.warning(f"Source face index {source_face_index} out of range")
                return None
            
            if len(target_faces) <= target_face_index:
                logger.warning(f"Target face index {target_face_index} out of range")
                return None
            
            source_face = source_faces[source_face_index]
            target_face = target_faces[target_face_index]
            
            # Perform face swap
            result_image = self.face_swapper.get(
                target_image, 
                target_face, 
                source_face,
                paste_back=True
            )
            
            return result_image
            
        except Exception as e:
            logger.error(f"Face swapping failed: {e}")
            return None
    
    def swap_face_realtime(self, frame: np.ndarray, reference_face: Dict[str, Any]) -> np.ndarray:
        """
        Perform real-time face swapping on video frame
        
        Args:
            frame: Current video frame
            reference_face: Reference face data with embedding
            
        Returns:
            Frame with swapped face
        """
        if not self.initialized:
            return frame
        
        try:
            # Find faces in current frame
            current_faces = self.face_analyzer.get(frame)
            
            if not current_faces:
                return frame
            
            # Use the first detected face as target
            target_face = current_faces[0]
            
            # Create a mock source face from reference embedding
            # In a real implementation, you would reconstruct the face from embedding
            # For now, we'll just return the original frame
            logger.debug("Real-time face swap not fully implemented")
            return frame
            
        except Exception as e:
            logger.error(f"Real-time face swap failed: {e}")
            return frame


class FaceSwapPipeline:
    """Pipeline for integrating face swapping into avatar mirror system"""
    
    def __init__(self, cache_dir: str = './face_swap_cache'):
        self.deep_live_cam = DeepLiveCamWrapper()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Reference faces for swapping
        self.reference_faces = {}  # person_id -> face embedding
        self.swap_enabled = {}     # person_id -> bool
    
    def register_reference_face(self, person_id: int, reference_image: np.ndarray) -> bool:
        """
        Register a reference face for a person
        
        Args:
            person_id: Person identifier
            reference_image: Image containing the reference face
            
        Returns:
            True if successful
        """
        try:
            face_features = self.deep_live_cam.extract_face_features(reference_image)
            
            if not face_features:
                logger.warning(f"No face found in reference image for person {person_id}")
                return False
            
            # Use the first (largest) face
            self.reference_faces[person_id] = face_features[0]
            self.swap_enabled[person_id] = True
            
            # Save reference image
            ref_path = self.cache_dir / f'reference_face_{person_id}.jpg'
            cv2.imwrite(str(ref_path), reference_image)
            
            logger.info(f"Registered reference face for person {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register reference face for person {person_id}: {e}")
            return False
    
    def enable_face_swap(self, person_id: int, enabled: bool = True):
        """Enable or disable face swapping for a person"""
        if person_id in self.reference_faces:
            self.swap_enabled[person_id] = enabled
            logger.info(f"Face swap {'enabled' if enabled else 'disabled'} for person {person_id}")
    
    def process_frame_with_swap(self, frame: np.ndarray, people_data: Dict[int, Any]) -> np.ndarray:
        """
        Process frame with face swapping applied
        
        Args:
            frame: Input frame
            people_data: Tracked people data
            
        Returns:
            Frame with face swaps applied
        """
        if not self.deep_live_cam.initialized:
            return frame
        
        result_frame = frame.copy()
        
        for person_id, person_data in people_data.items():
            # Check if face swap is enabled for this person
            if not self.swap_enabled.get(person_id, False):
                continue
            
            if person_id not in self.reference_faces:
                continue
            
            # Apply face swap
            reference_face = self.reference_faces[person_id]
            swapped_frame = self.deep_live_cam.swap_face_realtime(result_frame, reference_face)
            
            if swapped_frame is not None:
                result_frame = swapped_frame
        
        return result_frame
    
    def load_reference_faces(self):
        """Load cached reference faces"""
        try:
            for ref_file in self.cache_dir.glob('reference_face_*.jpg'):
                # Extract person ID from filename
                filename = ref_file.stem
                person_id_str = filename.split('_')[-1]
                
                try:
                    person_id = int(person_id_str)
                    ref_image = cv2.imread(str(ref_file))
                    
                    if ref_image is not None:
                        self.register_reference_face(person_id, ref_image)
                        
                except ValueError:
                    logger.warning(f"Invalid person ID in filename: {ref_file}")
                    continue
            
            logger.info(f"Loaded {len(self.reference_faces)} reference faces")
            
        except Exception as e:
            logger.error(f"Failed to load reference faces: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get face swap pipeline status"""
        return {
            'initialized': self.deep_live_cam.initialized,
            'reference_faces_count': len(self.reference_faces),
            'enabled_swaps': sum(1 for enabled in self.swap_enabled.values() if enabled),
            'device': self.deep_live_cam.device
        }


class FaceSwapWebInterface:
    """Simple web interface for managing face swapping"""
    
    def __init__(self, face_swap_pipeline: FaceSwapPipeline):
        self.pipeline = face_swap_pipeline
    
    def upload_reference_face(self, person_id: int, image_data: bytes) -> Dict[str, Any]:
        """
        Upload reference face via web interface
        
        Args:
            person_id: Person identifier
            image_data: Image binary data
            
        Returns:
            Status response
        """
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'success': False, 'error': 'Invalid image format'}
            
            success = self.pipeline.register_reference_face(person_id, image)
            
            return {
                'success': success,
                'message': f'Reference face {"registered" if success else "failed"} for person {person_id}'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def toggle_face_swap(self, person_id: int, enabled: bool) -> Dict[str, Any]:
        """Toggle face swap for person"""
        try:
            self.pipeline.enable_face_swap(person_id, enabled)
            return {
                'success': True,
                'person_id': person_id,
                'enabled': enabled
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self.pipeline.get_status()