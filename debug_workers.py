#!/usr/bin/env python3
"""
Debug worker thread processing
"""

import sys
import time
import threading
import queue
import logging
sys.path.insert(0, 'src')

from src.main import AvatarMirrorSystem
import cv2
import numpy as np

def test_worker_directly():
    """Test a worker thread directly"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🔧 Testing worker thread processing directly...")
    
    # Create system but don't run the full loop
    system = AvatarMirrorSystem()
    
    # Initialize just the components we need
    if not system.initialize_system():
        logger.error("Failed to initialize system")
        return False
    
    # Create a test work item
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(test_image, (256, 150), 50, (255, 255, 255), -1)  # Face
    cv2.rectangle(test_image, (200, 200), (312, 400), (128, 128, 128), -1)  # Body
    
    work_item = {
        'id': 999,
        'best_frame': test_image,
        'face_box': (206, 100, 100, 100),
        'timestamp': time.time()
    }
    
    logger.info("⚡ Processing work item directly (no queue)...")
    
    # Process directly without queue
    result = system._process_person(work_item)
    
    if result:
        logger.info("✅ Direct processing successful!")
        logger.info(f"Result: {result}")
        return True
    else:
        logger.error("❌ Direct processing failed")
        return False

def test_queue_system():
    """Test the queue system itself"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing queue system...")
    
    # Create simple queue test
    test_queue = queue.Queue()
    result_queue = queue.Queue()
    
    def simple_worker():
        logger.info("Worker: Started")
        try:
            item = test_queue.get(timeout=5.0)
            logger.info(f"Worker: Got item {item}")
            
            # Simulate processing
            time.sleep(1)
            result_queue.put(f"Processed {item}")
            test_queue.task_done()
            
            logger.info("Worker: Done")
        except queue.Empty:
            logger.error("Worker: Timeout")
        except Exception as e:
            logger.error(f"Worker: Error {e}")
    
    # Start worker
    worker_thread = threading.Thread(target=simple_worker)
    worker_thread.start()
    
    # Add work
    test_queue.put("test_item")
    logger.info("Main: Added test item to queue")
    
    # Wait for result
    try:
        result = result_queue.get(timeout=10.0)
        logger.info(f"Main: Got result: {result}")
        
        worker_thread.join(timeout=5.0)
        logger.info("✅ Queue system working correctly")
        return True
        
    except queue.Empty:
        logger.error("❌ No result received")
        return False

if __name__ == "__main__":
    print("🔍 Avatar Mirror Worker Debug\n")
    
    # Test 1: Simple queue system
    print("Test 1: Queue System")
    if test_queue_system():
        print("✅ Queue system works\n")
    else:
        print("❌ Queue system broken\n")
        sys.exit(1)
    
    # Test 2: Direct processing
    print("Test 2: Direct Processing")
    if test_worker_directly():
        print("✅ Direct processing works")
        print("\n🎯 Issue is with queue/threading communication")
    else:
        print("❌ Direct processing broken")
        print("\n🎯 Issue is with processing pipeline itself")
    
    print("\n💡 Diagnosis complete!")