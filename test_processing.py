#!/usr/bin/env python3
"""
Test processing pipeline without camera to isolate the issue
"""

import sys
import time
import logging
sys.path.insert(0, 'src')

from src.background_removal.processor import ImagePreprocessor
from src.reconstruction.pifuhd_direct import PIFuHDDirectPipeline
from src.rigging.unirig_wrapper import RiggingPipeline
from src.cache.cache_manager import CacheManager
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_processing_pipeline():
    """Test the processing pipeline with a dummy image"""
    
    logger.info("🧪 Testing Avatar Mirror processing pipeline...")
    
    # Create a dummy image (black square with white circle for face)
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(test_image, (256, 150), 50, (255, 255, 255), -1)  # White circle for face
    cv2.rectangle(test_image, (200, 200), (312, 400), (128, 128, 128), -1)  # Gray body
    
    face_box = (206, 100, 100, 100)  # x, y, w, h
    
    try:
        # Step 1: Test image preprocessing
        logger.info("📷 Step 1: Testing image preprocessing...")
        preprocessor = ImagePreprocessor()
        processed_image = preprocessor.prepare_for_reconstruction(test_image, face_box)
        logger.info("✅ Image preprocessing successful")
        
        # Step 2: Test 3D reconstruction
        logger.info("🎯 Step 2: Testing 3D reconstruction...")
        reconstruction_pipeline = PIFuHDDirectPipeline()
        
        reconstruction_data = {
            'id': 999,
            'best_frame': processed_image,
            'timestamp': time.time()
        }
        
        reconstruction_result = reconstruction_pipeline.process_person(reconstruction_data)
        
        if reconstruction_result.get('success', False):
            logger.info("✅ 3D reconstruction successful")
        else:
            logger.error(f"❌ 3D reconstruction failed: {reconstruction_result.get('error', 'Unknown')}")
            return False
        
        # Step 3: Test rigging
        logger.info("🦴 Step 3: Testing rigging...")
        rigging_pipeline = RiggingPipeline()
        rig_result = rigging_pipeline.process_reconstruction(reconstruction_result)
        
        if rig_result.get('success', False):
            logger.info("✅ Rigging successful")
        else:
            logger.error(f"❌ Rigging failed: {rig_result.get('error', 'Unknown')}")
            return False
        
        logger.info("🎉 Complete processing pipeline test successful!")
        return True
        
    except Exception as e:
        logger.error(f"💥 Processing pipeline test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_processing_pipeline()
    if success:
        print("\n🎉 Processing pipeline works! The issue is elsewhere.")
        print("Possible causes:")
        print("1. Person not getting confirmed in tracking")
        print("2. Quality threshold not being met") 
        print("3. Worker thread communication issue")
    else:
        print("\n❌ Processing pipeline has issues - this is the bottleneck!")
    
    sys.exit(0 if success else 1)