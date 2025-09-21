import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging
from PIL import Image
import io

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg not available, falling back to alternative methods")

try:
    from backgroundremover.bg import remove as bg_remove
    BACKGROUNDREMOVER_AVAILABLE = True
except ImportError:
    BACKGROUNDREMOVER_AVAILABLE = False
    logging.warning("backgroundremover not available")

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """Advanced background removal for human subjects"""
    
    def __init__(self, method: str = 'auto', model_name: str = 'u2net_human_seg'):
        """
        Initialize background remover
        
        Args:
            method: 'rembg', 'backgroundremover', 'opencv', or 'auto'
            model_name: Model for rembg (u2net_human_seg is best for people)
        """
        self.method = method
        self.model_name = model_name
        self.session = None
        
        if self.method == 'auto':
            self.method = self._choose_best_available_method()
        
        if self.method == 'rembg' and REMBG_AVAILABLE:
            self.session = new_session(model_name)
            logger.info(f"Initialized rembg with model: {model_name}")
        elif self.method == 'opencv':
            self._init_opencv_segmentation()
    
    def _choose_best_available_method(self) -> str:
        """Choose the best available background removal method"""
        if REMBG_AVAILABLE:
            return 'rembg'
        elif BACKGROUNDREMOVER_AVAILABLE:
            return 'backgroundremover'
        else:
            return 'opencv'
    
    def _init_opencv_segmentation(self):
        """Initialize OpenCV-based segmentation"""
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        logger.info("Initialized OpenCV background subtraction")
    
    def remove_background(self, image: np.ndarray, 
                         return_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Remove background from image
        
        Args:
            image: Input image as numpy array (BGR format)
            return_mask: Whether to return the mask along with the result
            
        Returns:
            Image with background removed, optionally with mask
        """
        if self.method == 'rembg':
            return self._remove_with_rembg(image, return_mask)
        elif self.method == 'backgroundremover':
            return self._remove_with_backgroundremover(image, return_mask)
        elif self.method == 'opencv':
            return self._remove_with_opencv(image, return_mask)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _remove_with_rembg(self, image: np.ndarray, 
                          return_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Remove background using rembg"""
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Remove background
        result = remove(pil_image, session=self.session)
        
        # Convert back to numpy array
        result_array = np.array(result)
        
        # Handle RGBA result
        if result_array.shape[2] == 4:
            # Extract alpha channel as mask
            mask = result_array[:, :, 3]
            # Convert RGB to BGR
            result_bgr = cv2.cvtColor(result_array[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            # Create mask from non-black pixels
            mask = np.any(result_array != [0, 0, 0], axis=2).astype(np.uint8) * 255
            result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        
        if return_mask:
            return result_bgr, mask
        return result_bgr
    
    def _remove_with_backgroundremover(self, image: np.ndarray, 
                                     return_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Remove background using backgroundremover"""
        # Convert to PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Remove background
        result = bg_remove(pil_image)
        result_array = np.array(result)
        
        # Handle RGBA result
        if result_array.shape[2] == 4:
            mask = result_array[:, :, 3]
            result_bgr = cv2.cvtColor(result_array[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            mask = np.any(result_array != [0, 0, 0], axis=2).astype(np.uint8) * 255
            result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        
        if return_mask:
            return result_bgr, mask
        return result_bgr
    
    def _remove_with_opencv(self, image: np.ndarray, 
                           return_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Remove background using OpenCV (simple method)"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to smooth edges
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        
        # Apply mask to image
        result = cv2.bitwise_and(image, image, mask=fg_mask)
        
        if return_mask:
            return result, fg_mask
        return result
    
    def refine_human_mask(self, image: np.ndarray, mask: np.ndarray, 
                         face_box: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Refine mask specifically for human subjects
        
        Args:
            image: Original image
            mask: Initial mask
            face_box: Optional face bounding box for better refinement
            
        Returns:
            Refined mask
        """
        refined_mask = mask.copy()
        
        # Ensure face region is included if provided
        if face_box:
            x, y, w, h = face_box
            # Expand face region slightly and ensure it's in the mask
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            refined_mask[y1:y2, x1:x2] = 255
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find largest contour (assumed to be the person)
            largest_contour = max(contours, key=cv2.contourArea)
            refined_mask.fill(0)
            cv2.fillPoly(refined_mask, [largest_contour], 255)
        
        # Apply Gaussian blur for smoother edges
        refined_mask = cv2.GaussianBlur(refined_mask, (3, 3), 0)
        
        return refined_mask
    
    def create_composite(self, foreground: np.ndarray, background: np.ndarray, 
                        mask: np.ndarray) -> np.ndarray:
        """
        Create composite image with custom background
        
        Args:
            foreground: Image with removed background
            background: New background image
            mask: Alpha mask
            
        Returns:
            Composite image
        """
        # Resize background to match foreground
        background_resized = cv2.resize(background, 
                                      (foreground.shape[1], foreground.shape[0]))
        
        # Normalize mask to 0-1 range
        mask_norm = mask.astype(np.float32) / 255.0
        
        # Create composite
        composite = (foreground * mask_norm[:, :, np.newaxis] + 
                    background_resized * (1 - mask_norm[:, :, np.newaxis]))
        
        return composite.astype(np.uint8)


class ImagePreprocessor:
    """Image preprocessing for 3D reconstruction"""
    
    def __init__(self):
        self.bg_remover = BackgroundRemover()
    
    def prepare_for_reconstruction(self, image: np.ndarray, 
                                 face_box: Tuple[int, int, int, int],
                                 target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Prepare image for 3D reconstruction
        
        Args:
            image: Input image
            face_box: Face bounding box for refinement
            target_size: Target image size for reconstruction
            
        Returns:
            Preprocessed image ready for PIFuHD
        """
        # Remove background
        clean_image, mask = self.bg_remover.remove_background(image, return_mask=True)
        
        # Refine mask for human subject
        refined_mask = self.bg_remover.refine_human_mask(image, mask, face_box)
        
        # Apply refined mask
        result = cv2.bitwise_and(image, image, mask=refined_mask)
        
        # Center and resize the person in the image
        result = self._center_and_resize_person(result, refined_mask, target_size)
        
        # Enhance image quality
        result = self._enhance_image_quality(result)
        
        return result
    
    def _center_and_resize_person(self, image: np.ndarray, mask: np.ndarray, 
                                target_size: Tuple[int, int]) -> np.ndarray:
        """Center and resize person in image"""
        # Find bounding box of person
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours, just resize
            return cv2.resize(image, target_size)
        
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 50
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Crop to person region
        cropped = image[y1:y2, x1:x2]
        
        # Resize while maintaining aspect ratio
        h_crop, w_crop = cropped.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w_crop, target_h / h_crop)
        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)
        
        resized = cv2.resize(cropped, (new_w, new_h))
        
        # Create centered result
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return result
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better reconstruction"""
        # Apply histogram equalization to improve contrast
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # Apply slight denoising
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Sharpen slightly
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel * 0.1)
        
        return sharpened