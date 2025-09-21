import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PersonData:
    """Data structure for tracking a person"""
    id: int
    last_seen: float
    face_history: deque
    best_frame: Optional[np.ndarray] = None
    best_quality_score: float = 0.0
    confirmed: bool = False
    processing_started: bool = False


class QualityAssessor:
    """Assess image quality for 3D reconstruction"""
    
    def __init__(self):
        self.min_face_size = 100
        self.min_body_visibility = 0.6
    
    def assess_frame_quality(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> float:
        """
        Assess quality of a frame for 3D reconstruction
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, width, height)
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        x, y, w, h = face_box
        height, width = image.shape[:2]
        
        quality_score = 0.0
        
        # Face size quality (larger is better, up to a point)
        face_size_score = min(1.0, max(0.0, (w * h - self.min_face_size**2) / (200**2)))
        quality_score += face_size_score * 0.3
        
        # Image sharpness (using Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_region = gray[y:y+h, x:x+w]
        sharpness = cv2.Laplacian(face_region, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 1000.0)  # Normalize
        quality_score += sharpness_score * 0.3
        
        # Body visibility estimation
        body_visibility_score = self._estimate_body_visibility(image, face_box)
        quality_score += body_visibility_score * 0.3
        
        # Lighting quality (avoid over/under exposure)
        lighting_score = self._assess_lighting(image, face_box)
        quality_score += lighting_score * 0.1
        
        return min(1.0, quality_score)
    
    def _estimate_body_visibility(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> float:
        """Estimate how much of the body is visible"""
        x, y, w, h = face_box
        height, width = image.shape[:2]
        
        # Estimate body region based on face position
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Expected body dimensions (rough estimates)
        estimated_shoulder_width = w * 2.5
        estimated_body_height = h * 6
        
        # Check if estimated body region fits in frame
        body_left = face_center_x - estimated_shoulder_width // 2
        body_right = face_center_x + estimated_shoulder_width // 2
        body_bottom = face_center_y + estimated_body_height // 2
        
        # Calculate visible body percentage
        visible_width = max(0, min(body_right, width) - max(body_left, 0))
        visible_height = max(0, min(body_bottom, height) - face_center_y)
        
        width_ratio = visible_width / estimated_shoulder_width
        height_ratio = visible_height / (estimated_body_height // 2)  # From face center down
        
        return min(1.0, (width_ratio + height_ratio) / 2)
    
    def _assess_lighting(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> float:
        """Assess lighting quality in the face region"""
        x, y, w, h = face_box
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Avoid very dark or very bright images
        dark_pixels = np.sum(hist[:50]) / np.sum(hist)
        bright_pixels = np.sum(hist[200:]) / np.sum(hist)
        
        # Good lighting should have most pixels in middle range
        if dark_pixels > 0.5 or bright_pixels > 0.3:
            return 0.3
        
        # Check for good contrast
        std_dev = np.std(gray_face)
        contrast_score = min(1.0, std_dev / 50.0)
        
        return contrast_score


class PersonTrackingSystem:
    """Advanced person tracking system with quality assessment"""
    
    def __init__(self, face_cache_size: int = 20):
        self.people: Dict[int, PersonData] = {}
        self.next_person_id = 0
        self.face_cache_size = face_cache_size
        self.quality_assessor = QualityAssessor()
        self.disappear_threshold = 3.0  # seconds
        self.confirmation_frames = 5  # frames needed to confirm person
        
    def update(self, image: np.ndarray, face_detections: List[Tuple[int, int, int, int, float]]) -> Dict[int, PersonData]:
        """
        Update tracking with new detections
        
        Args:
            image: Current frame
            face_detections: List of (x, y, w, h, confidence) detections
            
        Returns:
            Dictionary of currently tracked people
        """
        current_time = time.time()
        
        # Remove disappeared people
        self._cleanup_disappeared_people(current_time)
        
        if not face_detections:
            return self.people
        
        # Match detections to existing people
        matched_people, unmatched_detections = self._match_detections(
            face_detections, current_time
        )
        
        # Update matched people
        for person_id, detection in matched_people.items():
            self._update_person(person_id, image, detection, current_time)
        
        # Create new people for unmatched detections
        for detection in unmatched_detections:
            self._create_new_person(image, detection, current_time)
        
        return self.people
    
    def _cleanup_disappeared_people(self, current_time: float):
        """Remove people who haven't been seen recently"""
        to_remove = []
        for person_id, person in self.people.items():
            if current_time - person.last_seen > self.disappear_threshold:
                logger.info(f"Person {person_id} disappeared")
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.people[person_id]
    
    def _match_detections(self, detections: List[Tuple[int, int, int, int, float]], 
                         current_time: float) -> Tuple[Dict[int, Tuple], List[Tuple]]:
        """Match new detections to existing people"""
        if not self.people:
            return {}, detections
        
        # Calculate centroids for detections
        detection_centroids = []
        for x, y, w, h, conf in detections:
            centroid = (x + w//2, y + h//2)
            detection_centroids.append(centroid)
        
        # Get centroids of existing people (from their last known position)
        people_centroids = []
        people_ids = []
        for person_id, person in self.people.items():
            if person.face_history:
                last_face = person.face_history[-1]
                x, y, w, h = last_face[:4]
                centroid = (x + w//2, y + h//2)
                people_centroids.append(centroid)
                people_ids.append(person_id)
        
        if not people_centroids:
            return {}, detections
        
        # Calculate distance matrix
        distances = []
        for p_centroid in people_centroids:
            row = []
            for d_centroid in detection_centroids:
                distance = np.sqrt((p_centroid[0] - d_centroid[0])**2 + 
                                 (p_centroid[1] - d_centroid[1])**2)
                row.append(distance)
            distances.append(row)
        
        distances = np.array(distances)
        
        # Match using simple greedy algorithm (can be improved with Hungarian algorithm)
        max_distance = 100  # pixels
        matched_people = {}
        matched_detections = set()
        
        # Sort by distance and match
        for _ in range(min(len(people_ids), len(detections))):
            if distances.size == 0:
                break
                
            min_idx = np.unravel_index(distances.argmin(), distances.shape)
            person_idx, detection_idx = min_idx
            
            if distances[person_idx, detection_idx] < max_distance:
                person_id = people_ids[person_idx]
                matched_people[person_id] = detections[detection_idx]
                matched_detections.add(detection_idx)
                
                # Remove matched entries
                distances[person_idx, :] = np.inf
                distances[:, detection_idx] = np.inf
        
        # Get unmatched detections
        unmatched_detections = [
            det for i, det in enumerate(detections) 
            if i not in matched_detections
        ]
        
        return matched_people, unmatched_detections
    
    def _update_person(self, person_id: int, image: np.ndarray, 
                      detection: Tuple[int, int, int, int, float], 
                      current_time: float):
        """Update an existing person with new detection"""
        person = self.people[person_id]
        person.last_seen = current_time
        
        # Add to face history
        person.face_history.append(detection)
        if len(person.face_history) > self.confirmation_frames * 2:
            person.face_history.popleft()
        
        # Confirm person after enough detections
        if not person.confirmed and len(person.face_history) >= self.confirmation_frames:
            person.confirmed = True
            logger.info(f"Person {person_id} confirmed")
        
        # Assess frame quality and update best frame if needed
        if person.confirmed:
            x, y, w, h, conf = detection
            quality_score = self.quality_assessor.assess_frame_quality(image, (x, y, w, h))
            
            if quality_score > person.best_quality_score:
                person.best_quality_score = quality_score
                person.best_frame = image.copy()
                logger.info(f"Updated best frame for person {person_id} (quality: {quality_score:.3f})")
    
    def _create_new_person(self, image: np.ndarray, 
                          detection: Tuple[int, int, int, int, float], 
                          current_time: float):
        """Create a new person from detection"""
        person_id = self.next_person_id
        self.next_person_id += 1
        
        person = PersonData(
            id=person_id,
            last_seen=current_time,
            face_history=deque(maxlen=self.face_cache_size)
        )
        person.face_history.append(detection)
        
        self.people[person_id] = person
        logger.info(f"New person {person_id} detected")
    
    def get_people_ready_for_processing(self) -> List[PersonData]:
        """Get people who are ready for 3D reconstruction"""
        ready_people = []
        for person in self.people.values():
            if (person.confirmed and 
                person.best_frame is not None and 
                not person.processing_started and
                person.best_quality_score > 0.5):  # Minimum quality threshold
                ready_people.append(person)
        
        return ready_people
    
    def mark_person_processing_started(self, person_id: int):
        """Mark that processing has started for a person"""
        if person_id in self.people:
            self.people[person_id].processing_started = True
            logger.info(f"Started processing person {person_id}")