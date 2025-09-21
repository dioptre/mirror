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
from typing import Dict, Any, Optional
from pathlib import Path

# Import all our modules
try:
    # Try relative imports first (when run as module)
    from .face_detection.detector import UltraLightFaceDetector
    from .tracking.person_tracker import PersonTrackingSystem
    from .background_removal.processor import ImagePreprocessor
    from .reconstruction.pifuhd_wrapper import ReconstructionPipeline
    from .rigging.unirig_wrapper import RiggingPipeline
    from .pose_estimation.tempo_wrapper import PoseEstimationPipeline
    from .websocket.godot_client import GodotIntegration
    from .cache.cache_manager import CacheManager
    from .face_swap.deep_live_cam_wrapper import FaceSwapPipeline
    from .utils.config import config
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from face_detection.detector import UltraLightFaceDetector
    from tracking.person_tracker import PersonTrackingSystem
    from background_removal.processor import ImagePreprocessor
    from reconstruction.pifuhd_wrapper import ReconstructionPipeline
    from rigging.unirig_wrapper import RiggingPipeline
    from pose_estimation.tempo_wrapper import PoseEstimationPipeline
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
            self.reconstruction_pipeline = ReconstructionPipeline(
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
            self.pose_pipeline = PoseEstimationPipeline(streaming_enabled=True)
            
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
        
        # Update PIFuHD path
        pifuhd_path = self.model_manager.get_repository_path('pifuhd')
        if pifuhd_path.exists():
            self.config.set('reconstruction', 'pifuhd_path', str(pifuhd_path))
        
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
                
                # Process the work item
                result = self._process_person(work_item)
                
                # Put result in result queue
                if result:
                    self.result_queue.put(result)
                
                # Mark work as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")
        
        logger.info(f"{worker_name} stopped")
    
    def _process_person(self, person_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a person through the complete pipeline"""
        person_id = person_data['id']
        best_frame = person_data['best_frame']
        face_box = person_data.get('face_box')
        
        try:
            logger.info(f"Processing person {person_id}...")
            
            # Check if we already have a rigged model cached
            if self.cache_manager.has_rigged_model(person_id):
                logger.info(f"Person {person_id} already has rigged model in cache")
                cached_model = self.cache_manager.get_model_data(person_id)
                return {
                    'person_id': person_id,
                    'type': 'cached_model',
                    'data': cached_model
                }
            
            # Preprocess image (remove background, etc.)
            logger.debug(f"Preprocessing image for person {person_id}")
            processed_image = self.image_preprocessor.prepare_for_reconstruction(
                best_frame, face_box
            )
            
            # Run 3D reconstruction
            logger.debug(f"Running 3D reconstruction for person {person_id}")
            reconstruction_data = {
                'id': person_id,
                'best_frame': processed_image,
                'timestamp': time.time()
            }
            reconstruction_result = self.reconstruction_pipeline.process_person(reconstruction_data)
            
            if not reconstruction_result.get('success', False):
                logger.error(f"3D reconstruction failed for person {person_id}")
                return None
            
            # Cache reconstruction
            self.cache_manager.cache_reconstruction(person_id, reconstruction_result)
            
            # Run rigging
            logger.debug(f"Running rigging for person {person_id}")
            rig_result = self.rigging_pipeline.process_reconstruction(reconstruction_result)
            
            if not rig_result.get('success', False):
                logger.error(f"Rigging failed for person {person_id}")
                return None
            
            # Cache rigged model
            self.cache_manager.cache_rigged_model(person_id, rig_result)
            
            self.stats['models_created'] += 1
            logger.info(f"Successfully processed person {person_id} - model ready")
            
            return {
                'person_id': person_id,
                'type': 'new_model',
                'data': rig_result
            }
            
        except Exception as e:
            logger.error(f"Error processing person {person_id}: {e}")
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
                for person_data in ready_people:
                    # Mark as processing started to avoid duplicates
                    self.person_tracker.mark_person_processing_started(person_data.id)
                    
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
                        logger.info(f"Queued person {person_data.id} for processing")
                    except queue.Full:
                        logger.warning("Processing queue is full, skipping person")
                
                # Check for processing results
                self._handle_processing_results()
                
                # Run pose estimation for all tracked people
                pose_data = self.pose_pipeline.process_frame(frame, people)
                
                # Apply face swapping if enabled
                display_frame = frame
                if self.face_swap_pipeline:
                    display_frame = self.face_swap_pipeline.process_frame_with_swap(frame, people)
                
                # Show debug frame (optional)
                if logger.isEnabledFor(logging.DEBUG):
                    self._draw_debug_info(display_frame, face_detections, people)
                    cv2.imshow('Avatar Mirror Debug', display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit requested")
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
        # Draw face detections
        for detection in face_detections:
            x, y, w, h, conf = detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{conf:.2f}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw person IDs
        for person_id, person_data in people.items():
            if person_data.face_history:
                latest_face = person_data.face_history[-1]
                x, y, w, h = latest_face[:4]
                
                # Draw person ID
                cv2.putText(frame, f'Person {person_id}', (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw status
                status = "Ready" if person_data.confirmed else "Tracking"
                if person_data.processing_started:
                    status = "Processing"
                
                cv2.putText(frame, status, (x, y + h + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw system stats
        stats_text = f"Frames: {self.stats['frames_processed']} | People: {self.stats['people_detected']} | Models: {self.stats['models_created']}"
        cv2.putText(frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
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