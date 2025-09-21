"""
Fast reconstruction pipeline for real-time avatar creation
Uses simplified 3D reconstruction that's fast enough for real-time use
"""

import cv2
import numpy as np
import trimesh
import os
import logging
from typing import Dict, Any, Optional, Tuple
import time

logger = logging.getLogger(__name__)


class FastReconstruction:
    """Fast 3D reconstruction using depth estimation and mesh generation"""
    
    def __init__(self):
        self.initialized = True  # Always available
    
    def reconstruct_person(self, image: np.ndarray, output_path: str) -> Dict[str, Any]:
        """
        Fast person reconstruction using depth estimation
        
        Args:
            image: Preprocessed image (background removed)
            output_path: Where to save the mesh
            
        Returns:
            Reconstruction result
        """
        try:
            logger.info("🚀 Fast reconstruction starting...")
            
            # Step 1: Extract person silhouette
            silhouette = self._extract_silhouette(image)
            
            # Step 2: Estimate depth from silhouette  
            depth_map = self._estimate_depth_from_silhouette(silhouette)
            
            # Step 3: Generate 3D mesh from depth
            vertices, faces = self._generate_mesh_from_depth(depth_map, silhouette)
            
            # Step 4: Clean and smooth mesh
            vertices, faces = self._clean_mesh(vertices, faces)
            
            # Step 5: Save mesh
            mesh_path = self._save_mesh(vertices, faces, output_path)
            
            logger.info(f"✅ Fast reconstruction complete: {mesh_path}")
            
            return {
                'success': True,
                'mesh_path': mesh_path,
                'vertices': vertices,
                'faces': faces,
                'method': 'fast_reconstruction'
            }
            
        except Exception as e:
            logger.error(f"Fast reconstruction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        """Extract person silhouette from image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create binary mask (assuming background was already removed)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Clean up the silhouette
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _estimate_depth_from_silhouette(self, silhouette: np.ndarray) -> np.ndarray:
        """Estimate depth map from silhouette using distance transform"""
        # Use distance transform to estimate depth
        distance = cv2.distanceTransform(silhouette, cv2.DIST_L2, 5)
        
        # Normalize and scale
        distance = distance / np.max(distance) if np.max(distance) > 0 else distance
        
        # Create depth profile that's more human-like
        h, w = distance.shape
        
        # Create depth variation based on body structure
        depth_map = np.zeros_like(distance)
        
        for y in range(h):
            for x in range(w):
                if silhouette[y, x] > 0:
                    # Base depth from distance transform
                    base_depth = distance[y, x]
                    
                    # Add human body depth variation
                    # Head/torso area should be deeper
                    body_factor = 1.0
                    if y < h * 0.3:  # Head area
                        body_factor = 1.2
                    elif y < h * 0.7:  # Torso area
                        body_factor = 1.1
                    
                    # Side areas (arms) should be less deep
                    center_x = w // 2
                    side_distance = abs(x - center_x) / (w // 2)
                    if side_distance > 0.3:  # Arm areas
                        body_factor *= (1.0 - side_distance * 0.3)
                    
                    depth_map[y, x] = base_depth * body_factor * 0.3  # Scale to reasonable depth
        
        return depth_map
    
    def _generate_mesh_from_depth(self, depth_map: np.ndarray, 
                                  silhouette: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 3D mesh from depth map"""
        h, w = depth_map.shape
        
        # Create vertices from depth map
        vertices = []
        vertex_indices = {}
        
        # Scale factors to make human-sized mesh
        scale_x = 2.0 / w  # Map to 2-unit width
        scale_y = 3.0 / h  # Map to 3-unit height
        
        vertex_idx = 0
        for y in range(0, h, 4):  # Sample every 4 pixels for speed
            for x in range(0, w, 4):
                if silhouette[y, x] > 0:
                    # Convert to 3D coordinates
                    world_x = (x - w//2) * scale_x
                    world_y = (h - y - h//2) * scale_y  # Flip Y and center
                    world_z = depth_map[y, x]
                    
                    vertices.append([world_x, world_y, world_z])
                    vertex_indices[(x, y)] = vertex_idx
                    vertex_idx += 1
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Generate faces by connecting nearby vertices
        faces = []
        
        for y in range(0, h-4, 4):
            for x in range(0, w-4, 4):
                # Check if we have a 2x2 grid of vertices
                corners = [
                    (x, y), (x+4, y), (x, y+4), (x+4, y+4)
                ]
                
                # Get vertex indices for corners that exist
                corner_indices = []
                for corner in corners:
                    if corner in vertex_indices:
                        corner_indices.append(vertex_indices[corner])
                    else:
                        corner_indices.append(None)
                
                # Create triangles if we have enough vertices
                if len([idx for idx in corner_indices if idx is not None]) >= 3:
                    # Filter out None indices
                    valid_indices = [idx for idx in corner_indices if idx is not None]
                    
                    if len(valid_indices) >= 3:
                        # Create triangle(s)
                        faces.append([valid_indices[0], valid_indices[1], valid_indices[2]])
                        
                        if len(valid_indices) == 4:
                            # Create second triangle for quad
                            faces.append([valid_indices[0], valid_indices[2], valid_indices[3]])
        
        faces = np.array(faces, dtype=np.int32)
        
        logger.info(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces
    
    def _clean_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clean and smooth the mesh"""
        if len(vertices) == 0 or len(faces) == 0:
            return vertices, faces
        
        try:
            # Use trimesh for mesh cleanup
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Remove duplicate vertices
            mesh.merge_vertices()
            
            # Remove degenerate faces
            mesh.remove_degenerate_faces()
            
            # Smooth mesh slightly
            mesh = mesh.smoothed()
            
            return mesh.vertices, mesh.faces
            
        except Exception as e:
            logger.warning(f"Mesh cleaning failed: {e}, using original mesh")
            return vertices, faces
    
    def _save_mesh(self, vertices: np.ndarray, faces: np.ndarray, output_path: str) -> str:
        """Save mesh as OBJ file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        mesh_path = output_path + '.obj'
        
        with open(mesh_path, 'w') as f:
            f.write("# Fast reconstruction mesh\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                if len(face) >= 3:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        return mesh_path


class FastReconstructionPipeline:
    """Fast reconstruction pipeline for real-time use"""
    
    def __init__(self, cache_dir: str = './reconstruction_cache'):
        self.fast_recon = FastReconstruction()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("✅ Fast reconstruction pipeline initialized")
    
    def process_person(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process person using fast reconstruction"""
        person_id = person_data['id']
        image = person_data['best_frame']
        
        # Create output path
        output_path = os.path.join(self.cache_dir, f"person_{person_id}")
        
        # Run fast reconstruction
        result = self.fast_recon.reconstruct_person(image, output_path)
        
        # Add metadata
        result['person_id'] = person_id
        result['timestamp'] = person_data.get('timestamp')
        result['cache_path'] = output_path
        
        return result