#!/usr/bin/env python3
"""
Avatar Mirror System - Main Application
Real-time 3D avatar generation and pose tracking for mirror applications
"""

import cv2
import numpy as np
import time
import logging
import threading
import signal
import sys
import queue
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# Import all our modules
try:
    # Try relative imports first (when run as module)
    from .face_detection.detector import UltraLightFaceDetector
    from .tracking.person_tracker import PersonTrackingSystem
    from .background_removal.processor import ImagePreprocessor
    from .reconstruction.pifuhd_direct import PIFuHDDirectPipeline
    from .reconstruction.fast_reconstruction import FastReconstructionPipeline
    from .reconstruction.monoport_wrapper import MonoPortPipeline
    from .rigging.unirig_wrapper import RiggingPipeline
    from .pose_estimation.tempo_wrapper import PoseEstimationPipeline
    from .pose_estimation.lightweight_pose_wrapper import LightweightPoseEstimationPipeline
    from .pose_prediction.simlpe_wrapper import PosePredictionLayer
    from .websocket.godot_client import GodotIntegration
    from .cache.cache_manager import CacheManager
    from .face_swap.deep_live_cam_wrapper import FaceSwapPipeline
    from .models.model_manager import ModelManager
    from .utils.config import config
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from face_detection.detector import UltraLightFaceDetector
    from tracking.person_tracker import PersonTrackingSystem
    from background_removal.processor import ImagePreprocessor
    from reconstruction.pifuhd_direct import PIFuHDDirectPipeline
    from reconstruction.fast_reconstruction import FastReconstructionPipeline
    from reconstruction.monoport_wrapper import MonoPortPipeline
    from rigging.unirig_wrapper import RiggingPipeline
    from pose_estimation.tempo_wrapper import PoseEstimationPipeline
    from pose_estimation.lightweight_pose_wrapper import LightweightPoseEstimationPipeline
    from pose_prediction.simlpe_wrapper import PosePredictionLayer
    from websocket.godot_client import GodotIntegration
    from cache.cache_manager import CacheManager
    from face_swap.deep_live_cam_wrapper import FaceSwapPipeline
    from models.model_manager import ModelManager
    from utils.config import config

logger = logging.getLogger(__name__)


class AvatarMirrorSystem:
    """Main orchestrator for the Avatar Mirror system"""
    
    def __init__(self):
        # Initialize configuration
        self.config = config
        self._setup_logging()
        
        # System components
        self.face_detector = None
        self.person_tracker = None
        self.image_preprocessor = None
        self.reconstruction_pipeline = None
        self.rigging_pipeline = None
        self.pose_pipeline = None
        self.pose_prediction_layer = None
        self.godot_integration = None
        self.cache_manager = None
        self.face_swap_pipeline = None
        self.model_manager = None
        
        # Camera
        self.camera = None
        
        # Threading and queues
        self.processing_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.worker_threads = []
        self.running = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'people_detected': 0,
            'models_created': 0,
            'start_time': None
        }
        
        # Pose visualization data
        self.raw_pose_data = {}
        self.enhanced_pose_data = {}
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('logging', 'level', 'INFO'))
        log_file = self.config.get('logging', 'file', './avatar_mirror.log')
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info("Logging initialized")
    
    def initialize_system(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing Avatar Mirror System...")
            
            # Validate configuration
            config_issues = self.config.validate_config()
            if config_issues:
                logger.error("Configuration issues found:")
                for issue in config_issues:
                    logger.error(f"  - {issue}")
                return False
            
            logger.info("Configuration validated successfully")
            
            # Initialize model manager and download models
            logger.info("Initializing model manager...")
            self.model_manager = ModelManager(models_dir='./models')
            
            logger.info("🤖 Checking and downloading required AI models...")
            self._setup_models()
            
            # Initialize cache manager
            logger.info("Initializing cache manager...")
            cache_config = self.config.get('cache')
            self.cache_manager = CacheManager(
                cache_dir=cache_config['cache_dir'],
                use_redis=cache_config['use_redis']
            )
            
            # Initialize face detector
            logger.info("Initializing face detector...")
            face_config = self.config.get('face_detection')
            self.face_detector = UltraLightFaceDetector(
                model_path=face_config.get('model_path')
            )
            
            # Initialize person tracker
            logger.info("Initializing person tracker...")
            tracking_config = self.config.get('tracking')
            self.person_tracker = PersonTrackingSystem(
                face_cache_size=tracking_config['face_cache_size']
            )
            
            # Initialize image preprocessor
            logger.info("Initializing image preprocessor...")
            self.image_preprocessor = ImagePreprocessor()
            
            # Initialize reconstruction pipeline
            logger.info("Initializing 3D reconstruction pipeline...")
            recon_config = self.config.get('reconstruction')
            recon_method = recon_config.get('method', 'fast')
            
            if recon_method == 'pifuhd':
                logger.info("Using PIFuHD for high-quality reconstruction (slower)")
                self.reconstruction_pipeline = PIFuHDDirectPipeline(
                    cache_dir=str(Path(cache_config['cache_dir']) / 'reconstruction')
                )
            elif recon_method == 'monoport':
                logger.info("Using MonoPort for real-time human performance capture")
                from .reconstruction.monoport_wrapper import MonoPortPipeline
                self.reconstruction_pipeline = MonoPortPipeline(
                    cache_dir=str(Path(cache_config['cache_dir']) / 'reconstruction')
                )
            else:
                logger.info("Using fast reconstruction for real-time performance")
                self.reconstruction_pipeline = FastReconstructionPipeline(
                    cache_dir=str(Path(cache_config['cache_dir']) / 'reconstruction')
                )
            
            # Initialize rigging pipeline
            logger.info("Initializing rigging pipeline...")
            rig_config = self.config.get('rigging')
            self.rigging_pipeline = RiggingPipeline(
                cache_dir=str(Path(cache_config['cache_dir']) / 'rigging')
            )
            
            # Initialize pose estimation
            logger.info("Initializing pose estimation...")
            pose_config = self.config.get('pose_estimation')
            pose_method = pose_config.get('method', 'lightweight')
            
            if pose_method == 'lightweight':
                logger.info("Using lightweight 3D pose estimation (real-time)")
                from .pose_estimation.lightweight_pose_wrapper import LightweightPoseEstimationPipeline
                self.pose_pipeline = LightweightPoseEstimationPipeline(streaming_enabled=True)
            else:
                logger.info("Using TEMPO pose estimation (high-quality)")
                self.pose_pipeline = PoseEstimationPipeline(streaming_enabled=True)
            
            # Initialize pose prediction layer (siMLPe)
            logger.info("Initializing pose prediction layer...")
            pred_config = self.config.get('pose_prediction')
            pred_enabled = pred_config.get('enabled', True)
            
            if pred_enabled:
                logger.info("Using siMLPe for pose prediction and smoothing")
                from .pose_prediction.simlpe_wrapper import PosePredictionLayer
                self.pose_prediction_layer = PosePredictionLayer(
                    enabled=True,
                    prediction_horizon=pred_config.get('prediction_horizon', 3),
                    drift_config=pred_config
                )
            else:
                logger.info("Pose prediction disabled")
                self.pose_prediction_layer = None
            
            # Initialize Godot integration
            logger.info("Initializing Godot WebSocket integration...")
            ws_config = self.config.get('websocket')
            self.godot_integration = GodotIntegration(
                host=ws_config['host'],
                port=ws_config['port']
            )
            
            # Start Godot WebSocket server
            self.godot_integration.start()
            
            # Register pose callback
            self.pose_pipeline.register_pose_callback(self._on_pose_update)
            
            # Initialize face swapping (optional)
            face_swap_config = self.config.get('face_swap')
            if face_swap_config.get('enabled', False):
                logger.info("Initializing face swapping...")
                self.face_swap_pipeline = FaceSwapPipeline(
                    cache_dir=str(Path(cache_config['cache_dir']) / 'face_swap')
                )
                # Load any cached reference faces
                self.face_swap_pipeline.load_reference_faces()
            else:
                logger.info("Face swapping disabled")
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def _setup_models(self):
        """Setup and download required AI models"""
        try:
            # Check for basic models (face detection)
            basic_models = self.model_manager.check_and_download_models(['face_detection'])
            
            # Setup repositories
            logger.info("📦 Setting up AI model repositories...")
            repo_results = self.model_manager.setup_all_repositories()
            
            # Report results
            success_count = sum(repo_results.values())
            total_count = len(repo_results)
            
            if success_count == total_count:
                logger.info(f"✅ All {total_count} repositories set up successfully")
            else:
                logger.warning(f"⚠️ {success_count}/{total_count} repositories set up successfully")
                for repo, success in repo_results.items():
                    status = "✅" if success else "❌"
                    logger.info(f"  {status} {repo}")
            
            # Update component paths
            self._update_component_paths()
            
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            logger.info("System will continue with fallback models where available")
    
    def _update_component_paths(self):
        """Update component configurations with model paths"""
        # Update face detection model path
        face_model_path = self.model_manager.get_model_path(
            'face_detection', 'opencv_face_detector', 'opencv_face_detector_uint8.pb'
        )
        if face_model_path.exists():
            self.config.set('face_detection', 'model_path', str(face_model_path.parent))
        
        # Update PIFuHD path and checkpoint
        pifuhd_path = self.model_manager.get_repository_path('pifuhd')
        if pifuhd_path.exists():
            self.config.set('reconstruction', 'pifuhd_path', str(pifuhd_path))
            
            # Set checkpoint path if it exists
            checkpoint_path = pifuhd_path / 'checkpoints' / 'pifuhd.pt'
            if checkpoint_path.exists():
                self.config.set('reconstruction', 'checkpoint_path', str(checkpoint_path))
                logger.info(f"✅ PIFuHD model weights found: {checkpoint_path}")
            else:
                logger.warning("PIFuHD model weights not found")
        
        # Update UniRig path
        unirig_path = self.model_manager.get_repository_path('unirig')
        if unirig_path.exists():
            self.config.set('rigging', 'unirig_path', str(unirig_path))
        
        # Update TEMPO path
        tempo_path = self.model_manager.get_repository_path('tempo')
        if tempo_path.exists():
            self.config.set('pose_estimation', 'tempo_path', str(tempo_path))
        
        # Update Deep-Live-Cam path
        dlc_path = self.model_manager.get_repository_path('deep_live_cam')
        if dlc_path.exists():
            self.config.set('face_swap', 'deep_live_cam_path', str(dlc_path))
        
        logger.info("✅ Component paths updated with downloaded models")
    
    def initialize_camera(self) -> bool:
        """Initialize camera"""
        try:
            camera_config = self.config.get('camera')
            
            logger.info(f"Initializing camera {camera_config['device_id']}...")
            self.camera = cv2.VideoCapture(camera_config['device_id'])
            
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
            self.camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def start_processing_workers(self):
        """Start background processing worker threads"""
        processing_config = self.config.get('processing')
        num_workers = processing_config['num_workers']
        
        logger.info(f"Starting {num_workers} processing workers...")
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._processing_worker,
                args=(f"Worker-{i}",),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info("Processing workers started")
    
    def _processing_worker(self, worker_name: str):
        """Background processing worker"""
        logger.info(f"{worker_name} started")
        
        while self.running:
            try:
                # Get work item
                work_item = self.processing_queue.get(timeout=1.0)
                
                if work_item is None:  # Shutdown signal
                    break
                
                logger.info(f"{worker_name}: Got work item for person {work_item.get('id', 'unknown')}")
                
                # Process the work item with timeout protection
                try:
                    result = self._process_person(work_item)
                    
                    # Put result in result queue
                    if result:
                        self.result_queue.put(result)
                        logger.info(f"{worker_name}: Processing completed successfully")
                    else:
                        logger.warning(f"{worker_name}: Processing returned no result")
                        
                except Exception as proc_error:
                    logger.error(f"{worker_name}: Processing error: {proc_error}")
                    import traceback
                    logger.error(f"{worker_name}: Traceback: {traceback.format_exc()}")
                
                # Mark work as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")
                # Still mark task as done to prevent queue from backing up
                try:
                    self.processing_queue.task_done()
                except:
                    pass
        
        logger.info(f"{worker_name} stopped")
    
    def _process_person(self, person_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a person through the complete pipeline"""
        person_id = person_data['id']
        best_frame = person_data['best_frame']
        face_box = person_data.get('face_box')
        
        try:
            logger.info(f"🔄 Starting processing for person {person_id}...")
            
            # Check if we already have a rigged model cached
            if self.cache_manager.has_rigged_model(person_id):
                logger.info(f"Person {person_id} already has rigged model in cache")
                cached_model = self.cache_manager.get_model_data(person_id)
                return {
                    'person_id': person_id,
                    'type': 'cached_model',
                    'data': cached_model
                }
            
            # Step 1: Preprocess image (remove background, etc.)
            logger.info(f"📷 Step 1/3: Preprocessing image for person {person_id}")
            processed_image = self.image_preprocessor.prepare_for_reconstruction(
                best_frame, face_box
            )
            logger.info(f"✅ Image preprocessing complete for person {person_id}")
            
            # Step 2: Run 3D reconstruction
            logger.info(f"🎯 Step 2/3: Running 3D reconstruction for person {person_id}")
            reconstruction_data = {
                'id': person_id,
                'best_frame': processed_image,
                'timestamp': time.time()
            }
            reconstruction_result = self.reconstruction_pipeline.process_person(reconstruction_data)
            
            if not reconstruction_result.get('success', False):
                logger.error(f"❌ 3D reconstruction failed for person {person_id}: {reconstruction_result.get('error', 'Unknown error')}")
                return None
            
            logger.info(f"✅ 3D reconstruction complete for person {person_id}")
            
            # Cache reconstruction
            self.cache_manager.cache_reconstruction(person_id, reconstruction_result)
            
            # Step 3: Run rigging
            logger.info(f"🦴 Step 3/3: Running rigging for person {person_id}")
            rig_result = self.rigging_pipeline.process_reconstruction(reconstruction_result)
            
            if not rig_result.get('success', False):
                logger.error(f"❌ Rigging failed for person {person_id}: {rig_result.get('error', 'Unknown error')}")
                return None
            
            logger.info(f"✅ Rigging complete for person {person_id}")
            
            # Cache rigged model
            self.cache_manager.cache_rigged_model(person_id, rig_result)
            
            self.stats['models_created'] += 1
            logger.info(f"🎉 Successfully processed person {person_id} - model ready!")
            
            return {
                'person_id': person_id,
                'type': 'new_model',
                'data': rig_result
            }
            
        except Exception as e:
            logger.error(f"💥 Critical error processing person {person_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _on_pose_update(self, pose_data: Dict[int, Dict[str, Any]]):
        """Handle pose updates from pose estimation pipeline"""
        # Forward pose data to Godot
        if self.godot_integration:
            self.godot_integration.update_poses(pose_data)
    
    def run(self):
        """Main application loop"""
        if not self.initialize_system():
            logger.error("System initialization failed")
            return False
        
        if not self.initialize_camera():
            logger.error("Camera initialization failed")
            return False
        
        # Start processing workers
        self.start_processing_workers()
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        logger.info("Avatar Mirror System started - processing frames...")
        
        try:
            frame_count = 0
            last_stats_time = time.time()
            
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                frame_count += 1
                self.stats['frames_processed'] = frame_count
                
                # Detect faces
                face_detections = self.face_detector.detect_faces(frame)
                
                # Update person tracking
                people = self.person_tracker.update(frame, face_detections)
                
                # Check for new people ready for processing
                ready_people = self.person_tracker.get_people_ready_for_processing()
                
                # Debug: Log tracking status
                if self.stats['frames_processed'] % 30 == 0:  # Every 30 frames (~1 second)
                    if people:
                        for person_id, person_data in people.items():
                            if hasattr(person_data, 'face_history'):
                                history_len = len(person_data.face_history)
                                confirmed = getattr(person_data, 'confirmed', False)
                                quality = getattr(person_data, 'best_quality_score', 0.0)
                                processing = getattr(person_data, 'processing_started', False)
                                
                                logger.info(f"Person {person_id}: confirmed={confirmed}, quality={quality:.2f}, "
                                          f"history={history_len}, processing={processing}")
                    else:
                        logger.debug("No people currently tracked")
                
                for person_data in ready_people:
                    # Mark as processing started to avoid duplicates
                    self.person_tracker.mark_person_processing_started(person_data.id)
                    
                    logger.info(f"🚀 Person {person_data.id} ready for processing! Quality: {person_data.best_quality_score:.2f}")
                    
                    # Queue for background processing
                    try:
                        processing_data = {
                            'id': person_data.id,
                            'best_frame': person_data.best_frame.copy(),
                            'face_box': self._get_latest_face_box(person_data),
                            'timestamp': time.time()
                        }
                        self.processing_queue.put(processing_data, block=False)
                        self.stats['people_detected'] += 1
                        logger.info(f"✅ Queued person {person_data.id} for 3D reconstruction")
                    except queue.Full:
                        logger.warning("Processing queue is full, skipping person")
                
                # Check for processing results
                self._handle_processing_results()
                
                # Run pose estimation for all tracked people
                raw_pose_data = self.pose_pipeline.process_frame(frame, people)
                
                # Apply pose prediction and smoothing layer
                if self.pose_prediction_layer:
                    enhanced_pose_data = self.pose_prediction_layer.process_poses(raw_pose_data)
                else:
                    enhanced_pose_data = raw_pose_data
                
                # Store both raw and enhanced pose data for visualization
                pose_data = enhanced_pose_data
                self.raw_pose_data = raw_pose_data  # Store for visualization
                self.enhanced_pose_data = enhanced_pose_data
                
                # Apply face swapping if enabled
                display_frame = frame
                if self.face_swap_pipeline:
                    display_frame = self.face_swap_pipeline.process_frame_with_swap(frame, people)
                
                # Show debug frame (always show for now to help with debugging)
                self._draw_debug_info(display_frame, face_detections, people)
                cv2.imshow('Avatar Mirror Debug', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('d') and self.pose_prediction_layer:
                    # Manual drift trigger for testing
                    logger.info("🎭 Manual drift trigger activated!")
                    for person_id in self.pose_prediction_layer.person_drift_data:
                        if self.pose_prediction_layer.person_drift_data[person_id].state.value == 'following':
                            self.pose_prediction_layer._start_drift(person_id, time.time())
                            break
                
                # Print stats periodically
                current_time = time.time()
                if current_time - last_stats_time >= 10.0:  # Every 10 seconds
                    self._print_stats()
                    last_stats_time = current_time
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            self._shutdown()
        
        return True
    
    def _get_latest_face_box(self, person_data) -> tuple:
        """Get latest face bounding box for person"""
        if person_data.face_history:
            latest_detection = person_data.face_history[-1]
            return latest_detection[:4]  # x, y, w, h
        return (0, 0, 100, 100)  # Default fallback
    
    def _handle_processing_results(self):
        """Handle results from background processing"""
        while True:
            try:
                result = self.result_queue.get_nowait()
                
                if result['type'] == 'new_model':
                    # Notify Godot about new character
                    self.godot_integration.notify_new_character(
                        result['person_id'], 
                        result['data']
                    )
                    logger.info(f"Notified Godot about new character: person {result['person_id']}")
                
            except queue.Empty:
                break
    
    def _draw_debug_info(self, frame, face_detections, people):
        """Draw debugging information on frame"""
        h, w = frame.shape[:2]
        
        # Draw face detections with more detail
        for i, detection in enumerate(face_detections):
            x, y, w_face, h_face, conf = detection
            
            # Draw face rectangle
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), color, 2)
            cv2.putText(frame, f'Face {i}: {conf:.2f}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw person tracking info with more detail  
        for person_id, person_data in people.items():
            # Debug: Print what type of object person_data is (use frames_processed instead of frame_count)
            if self.stats['frames_processed'] % 60 == 0:  # Every 2 seconds
                logger.info(f"Debug: person_data type for {person_id}: {type(person_data)}")
                if hasattr(person_data, '__dict__'):
                    logger.info(f"Debug: person_data attributes: {list(person_data.__dict__.keys())}")
            
            if hasattr(person_data, 'face_history') and person_data.face_history:
                latest_face = person_data.face_history[-1]
                x, y, w_face, h_face = latest_face[:4]
                
                # Draw person bounding box
                cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (255, 0, 0), 3)
                
                # Draw person ID
                cv2.putText(frame, f'Person {person_id}', (x, y - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw detailed status
                status_lines = []
                
                # Check confirmed status
                confirmed = getattr(person_data, 'confirmed', False)
                if confirmed:
                    status_lines.append("✓ Confirmed")
                else:
                    history_len = len(person_data.face_history) if hasattr(person_data, 'face_history') else 0
                    status_lines.append(f"Tracking ({history_len}/5)")
                
                # Check processing status
                processing_started = getattr(person_data, 'processing_started', False)
                if processing_started:
                    status_lines.append("🔄 Processing")
                else:
                    quality_score = getattr(person_data, 'best_quality_score', 0.0)
                    status_lines.append(f"Quality: {quality_score:.2f}")
                    
                    # Show if ready for processing
                    best_frame = getattr(person_data, 'best_frame', None)
                    if confirmed and best_frame is not None and quality_score > 0.5:
                        status_lines.append("🚀 READY!")
                    elif not confirmed:
                        status_lines.append("Need more frames")
                    elif quality_score <= 0.5:
                        status_lines.append("Quality too low")
                
                # Draw status lines
                for i, status in enumerate(status_lines):
                    cv2.putText(frame, status, (x, y + h_face + 20 + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Check if person meets visibility requirements
                if hasattr(self.face_detector, 'validate_person_visibility'):
                    is_valid = self.face_detector.validate_person_visibility(frame, (x, y, w_face, h_face))
                    visibility_color = (0, 255, 0) if is_valid else (0, 0, 255)
                    visibility_text = "✓ Visible" if is_valid else "✗ Not visible"
                    cv2.putText(frame, visibility_text, (x, y + h_face + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, visibility_color, 1)
        
        # Draw clean pose skeletons (stick figures only)
        self._draw_clean_pose_skeletons(frame)
        
        # Draw comprehensive system stats
        stats_lines = [
            f"Frames: {self.stats['frames_processed']}",
            f"People Detected: {self.stats['people_detected']}",
            f"Models Created: {self.stats['models_created']}",
            f"Active People: {len(people)}",
            f"Face Detections: {len(face_detections)}",
        ]
        
        # Add Godot connection status
        if self.godot_integration:
            client_count = self.godot_integration.get_client_count()
            stats_lines.append(f"Godot Clients: {client_count}")
        
        # Add processing queue status
        queue_size = self.processing_queue.qsize() if hasattr(self, 'processing_queue') else 0
        stats_lines.append(f"Processing Queue: {queue_size}")
        
        # Add pose prediction stats
        if self.pose_prediction_layer:
            pred_stats = self.pose_prediction_layer.get_prediction_stats()
            stats_lines.append(f"Pose Prediction: {'ON' if pred_stats['enabled'] else 'OFF'}")
            if pred_stats['enabled']:
                stats_lines.append(f"Tracked Histories: {pred_stats['tracked_people']}")
                
                # Add drift stats
                drift_stats = pred_stats.get('drift_stats', {})
                if drift_stats:
                    active_drifts = sum(1 for state in drift_stats.values() if state != 'following')
                    stats_lines.append(f"Drift Active: {active_drifts}/{len(drift_stats)}")
        
        # Draw stats with background
        stats_bg_height = len(stats_lines) * 25 + 20
        cv2.rectangle(frame, (10, 10), (400, stats_bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, stats_bg_height), (255, 255, 255), 1)
        
        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (20, 35 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit, 'd' to trigger drift",
            "Stand in front of camera for avatar creation",
            "Green=Real pose, Blue=Avatar, Red=Drifting"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 80 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw pose legend
        legend_y_start = h - 160
        legend_items = [
            ("Green: Real Human Pose", (0, 255, 0)),
            ("Blue: Avatar Following", (255, 0, 0)),
            ("Red: Avatar Drifting", (0, 0, 255)),
            ("Orange: Avatar Returning", (0, 165, 255))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y_start + i * 20
            # Draw color indicator
            cv2.rectangle(frame, (20, y_pos - 10), (35, y_pos + 5), color, -1)
            # Draw text
            cv2.putText(frame, text, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_pose_skeletons(self, frame):
        """Draw pose skeletons showing both raw and drift-enhanced poses"""
        h, w = frame.shape[:2]
        
        # Define skeleton connections
        skeleton_connections = [
            ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        
        # Draw pose skeletons for each person
        for person_id in self.raw_pose_data.keys():
            # Draw raw pose (green - what human is actually doing)
            raw_pose = self.raw_pose_data.get(person_id, {})
            if raw_pose.get('pose_present', False):
                self._draw_skeleton(frame, raw_pose, color=(0, 255, 0), 
                                  label=f"Real {person_id}", connections=skeleton_connections)
            
            # Draw enhanced/drift pose (blue/red - what avatar is doing)
            enhanced_pose = self.enhanced_pose_data.get(person_id, {})
            if enhanced_pose.get('pose_present', False):
                drift_state = enhanced_pose.get('drift_state', 'following')
                
                # Color based on drift state
                if drift_state == 'following':
                    color = (255, 0, 0)  # Blue for normal following
                    label = f"Avatar {person_id}"
                elif drift_state in ['drifting', 'drift_start']:
                    color = (0, 0, 255)  # Red for drifting
                    label = f"Drift {person_id}"
                elif drift_state == 'returning':
                    color = (0, 165, 255)  # Orange for returning
                    label = f"Return {person_id}"
                else:
                    color = (128, 128, 255)  # Light blue for cooldown
                    label = f"Cool {person_id}"
                
                self._draw_skeleton(frame, enhanced_pose, color=color,
                                  label=label, connections=skeleton_connections)
    
    def _draw_skeleton(self, frame, pose_data: Dict[str, Any], color: Tuple[int, int, int], 
                      label: str, connections: List[Tuple[str, str]]):
        """Draw a pose skeleton on the frame"""
        if not pose_data.get('pose_present', False):
            return
        
        joints = pose_data.get('joints', {})
        h, w = frame.shape[:2]
        
        # Debug: Log joint positions for first few frames
        if self.stats['frames_processed'] % 100 == 0:  # Every ~3 seconds
            logger.debug(f"Pose joints for {label}: {list(joints.keys())}")
            if 'nose' in joints:
                logger.debug(f"Nose position: {joints['nose']['position']}")
        
        # Convert pose coordinates to pixel coordinates
        joint_pixels = {}
        for joint_name, joint_data in joints.items():
            pos = joint_data.get('position', [0, 0, 0])
            confidence = joint_data.get('confidence', 0.0)
            
            # Skip low confidence joints
            if confidence < 0.3:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            if 0.0 <= pos[0] <= 1.0 and 0.0 <= pos[1] <= 1.0:  # Normalized coordinates [0,1]
                pixel_x = int(pos[0] * w)
                pixel_y = int(pos[1] * h)
            elif abs(pos[0]) <= 2.0 and abs(pos[1]) <= 2.0:  # Normalized coordinates [-1,1] or [-2,2]
                pixel_x = int((pos[0] + 1.0) * w / 2.0)
                pixel_y = int((pos[1] + 1.0) * h / 2.0)
            else:  # Already pixel coordinates
                pixel_x = int(pos[0])
                pixel_y = int(pos[1])
            
            # Keep within frame bounds
            pixel_x = max(5, min(w-5, pixel_x))
            pixel_y = max(5, min(h-5, pixel_y))
            
            joint_pixels[joint_name] = (pixel_x, pixel_y)
            
            # Draw joint point (larger for visibility)
            cv2.circle(frame, (pixel_x, pixel_y), 6, color, -1)
            cv2.circle(frame, (pixel_x, pixel_y), 8, (255, 255, 255), 1)  # White outline
        
        # Draw skeleton connections (stick figure lines)
        for joint1, joint2 in connections:
            if joint1 in joint_pixels and joint2 in joint_pixels:
                pt1 = joint_pixels[joint1]
                pt2 = joint_pixels[joint2]
                
                # Check if both joints exist and have reasonable confidence
                conf1 = joints.get(joint1, {}).get('confidence', 0.0)
                conf2 = joints.get(joint2, {}).get('confidence', 0.0)
                
                if conf1 > 0.3 and conf2 > 0.3:
                    # Draw thick line for visibility
                    cv2.line(frame, pt1, pt2, color, 3)
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 1)  # White outline
        
        # Draw label near the pose
        if joint_pixels.get('nose'):
            nose_pos = joint_pixels['nose']
            label_pos = (nose_pos[0] + 20, nose_pos[1] - 20)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, 
                         (label_pos[0] - 2, label_pos[1] - label_size[1] - 2),
                         (label_pos[0] + label_size[0] + 2, label_pos[1] + 2),
                         (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add drift state indicator
            drift_state = pose_data.get('drift_state', 'unknown')
            if drift_state != 'following':
                state_label = f"[{drift_state.upper()}]"
                state_pos = (label_pos[0], label_pos[1] + 15)
                cv2.putText(frame, state_label, state_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_clean_pose_skeletons(self, frame):
        """Draw clean stick figure pose skeletons without other debug elements"""
        if not hasattr(self, 'raw_pose_data') or not hasattr(self, 'enhanced_pose_data'):
            return
        
        h, w = frame.shape[:2]
        
        # Simple skeleton connections for clean stick figure
        stick_connections = [
            # Head to torso
            ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
            # Shoulders
            ('left_shoulder', 'right_shoulder'),
            # Arms
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            # Torso
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            # Legs  
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        
        # Draw for each tracked person
        for person_id in self.enhanced_pose_data.keys():
            # Get pose data
            raw_pose = self.raw_pose_data.get(person_id, {})
            enhanced_pose = self.enhanced_pose_data.get(person_id, {})
            
            # Draw real human pose (green stick figure)
            if raw_pose.get('pose_present', False):
                self._draw_clean_stick_figure(frame, raw_pose, (0, 255, 0), stick_connections, "Real")
            
            # Draw avatar pose (colored based on drift state)
            if enhanced_pose.get('pose_present', False):
                drift_state = enhanced_pose.get('drift_state', 'following')
                
                if drift_state == 'following':
                    color = (255, 100, 0)  # Bright blue
                    label = "Following"
                elif drift_state in ['drifting', 'drift_start']:
                    color = (0, 50, 255)  # Bright red
                    label = "DRIFTING"
                elif drift_state == 'returning':
                    color = (0, 165, 255)  # Orange
                    label = "Returning"
                else:
                    color = (180, 180, 255)  # Light purple for cooldown
                    label = "Cooldown"
                
                self._draw_clean_stick_figure(frame, enhanced_pose, color, stick_connections, label)
    
    def _draw_clean_stick_figure(self, frame, pose_data: Dict[str, Any], color: Tuple[int, int, int], 
                                connections: List[Tuple[str, str]], label: str):
        """Draw a clean stick figure without extra elements"""
        if not pose_data.get('pose_present', False):
            return
        
        joints = pose_data.get('joints', {})
        h, w = frame.shape[:2]
        
        # Convert joints to pixel coordinates
        stick_joints = {}
        
        for joint_name, joint_data in joints.items():
            pos = joint_data.get('position', [0, 0, 0])
            confidence = joint_data.get('confidence', 0.0)
            
            if confidence < 0.4:  # Higher confidence threshold for clean display
                continue
            
            # Handle different coordinate systems
            if 0.0 <= pos[0] <= 1.0 and 0.0 <= pos[1] <= 1.0:
                # Normalized [0,1] coordinates
                x = int(pos[0] * w)
                y = int(pos[1] * h)
            elif -1.0 <= pos[0] <= 1.0 and -1.0 <= pos[1] <= 1.0:
                # Normalized [-1,1] coordinates  
                x = int((pos[0] + 1.0) * w / 2.0)
                y = int((pos[1] + 1.0) * h / 2.0)
            else:
                # Pixel coordinates
                x = int(pos[0])
                y = int(pos[1])
            
            # Keep in bounds
            x = max(10, min(w-10, x))
            y = max(10, min(h-10, y))
            
            stick_joints[joint_name] = (x, y)
        
        # Draw stick figure connections
        for joint1, joint2 in connections:
            if joint1 in stick_joints and joint2 in stick_joints:
                pt1 = stick_joints[joint1]
                pt2 = stick_joints[joint2]
                
                # Draw clean line (no triangles, just lines!)
                cv2.line(frame, pt1, pt2, color, 3)  # Thick colored line
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1)  # Thin white outline
        
        # Draw joint dots
        for joint_name, (x, y) in stick_joints.items():
            cv2.circle(frame, (x, y), 5, color, -1)  # Filled circle
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)  # White outline
        
        # Draw simple label (top-right corner)
        if stick_joints:  # Only if we have joints to draw
            # Find a representative person_id from the current loop context
            person_ids = list(self.enhanced_pose_data.keys())
            if person_ids:
                current_person_id = person_ids[0]  # Use first person for label
                label_y = 30 + len([p for p in person_ids if p <= current_person_id]) * 20
                cv2.putText(frame, f"{label}", (w - 150, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _print_stats(self):
        """Print system statistics"""
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            fps = self.stats['frames_processed'] / uptime if uptime > 0 else 0
            
            logger.info("=== Avatar Mirror Stats ===")
            logger.info(f"Uptime: {uptime:.1f}s")
            logger.info(f"Frames processed: {self.stats['frames_processed']} ({fps:.1f} FPS)")
            logger.info(f"People detected: {self.stats['people_detected']}")
            logger.info(f"Models created: {self.stats['models_created']}")
            logger.info(f"Godot clients: {self.godot_integration.get_status()['client_count']}")
            
            # Cache stats
            cache_stats = self.cache_manager.get_cache_stats()
            logger.info(f"Cache: {cache_stats['face_cache_size']} faces, {cache_stats['model_cache_size']} models")
            
            # Face swap stats
            if self.face_swap_pipeline:
                swap_stats = self.face_swap_pipeline.get_status()
                logger.info(f"Face swap: {swap_stats['reference_faces_count']} references, {swap_stats['enabled_swaps']} active")
            
            logger.info("==========================")
    
    def _shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down Avatar Mirror System...")
        
        # Stop main loop
        self.running = False
        
        # Stop camera
        if self.camera:
            self.camera.release()
        
        # Stop processing workers
        logger.info("Stopping processing workers...")
        for _ in self.worker_threads:
            self.processing_queue.put(None)  # Send shutdown signal
        
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        # Stop Godot integration
        if self.godot_integration:
            self.godot_integration.stop()
        
        # Cleanup cache
        if self.cache_manager:
            self.cache_manager.cleanup_cache()
        
        # Close CV2 windows
        cv2.destroyAllWindows()
        
        logger.info("System shutdown complete")


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    sys.exit(0)


def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run the system
    system = AvatarMirrorSystem()
    
    try:
        success = system.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()