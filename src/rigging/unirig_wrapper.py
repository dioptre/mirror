import os
import sys
import numpy as np
import trimesh
import json
import subprocess
import tempfile
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class UniRigWrapper:
    """Wrapper for UniRig automatic rigging system"""
    
    def __init__(self, unirig_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize UniRig wrapper
        
        Args:
            unirig_path: Path to UniRig installation
            device: Device to use for processing
        """
        self.device = device
        self.unirig_path = unirig_path or self._find_unirig_installation()
        self.initialized = False
        
        if self.unirig_path and os.path.exists(self.unirig_path):
            self._setup_unirig()
        else:
            logger.warning("UniRig not found. Will attempt to download on first use.")
    
    def _find_unirig_installation(self) -> Optional[str]:
        """Try to find existing UniRig installation"""
        possible_paths = [
            './models/rigging/UniRig',
            './UniRig',
            '../UniRig', 
            '../../UniRig',
            os.path.expanduser('~/UniRig'),
            '/opt/UniRig'
        ]
        
        for path in possible_paths:
            # Check for any UniRig indicator files
            if (os.path.exists(os.path.join(path, 'unirig')) or
                os.path.exists(os.path.join(path, 'README.md')) or
                os.path.exists(os.path.join(path, 'src'))):
                logger.info(f"Found UniRig at {path}")
                return path
        
        return None
    
    def _setup_unirig(self):
        """Setup UniRig environment"""
        try:
            # Add UniRig to Python path
            if self.unirig_path not in sys.path:
                sys.path.insert(0, self.unirig_path)
            
            self.initialized = True
            logger.info("UniRig environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup UniRig: {e}")
            self.initialized = False
    
    def download_unirig(self, install_path: str = './UniRig') -> bool:
        """
        Download and setup UniRig
        
        Args:
            install_path: Where to install UniRig
            
        Returns:
            True if successful
        """
        try:
            logger.info("Downloading UniRig...")
            
            # Clone repository
            subprocess.run([
                'git', 'clone',
                'https://github.com/VAST-AI-Research/UniRig.git',
                install_path
            ], check=True)
            
            self.unirig_path = install_path
            self._setup_unirig()
            
            return self.initialized
            
        except Exception as e:
            logger.error(f"Failed to download UniRig: {e}")
            return False
    
    def rig_mesh(self, mesh_path: str, output_path: str, 
                 rig_type: str = 'humanoid') -> Dict[str, Any]:
        """
        Rig a 3D mesh with skeleton
        
        Args:
            mesh_path: Path to input mesh file
            output_path: Path for rigged output
            rig_type: Type of rig to create ('humanoid', 'quadruped', etc.)
            
        Returns:
            Rigging results dictionary
        """
        if not self.initialized:
            if not self.download_unirig():
                raise RuntimeError("Failed to setup UniRig")
        
        try:
            # Load mesh
            mesh = trimesh.load_mesh(mesh_path)
            
            # Run rigging process
            rig_result = self._run_rigging_process(mesh, rig_type)
            
            # Save rigged mesh
            self._save_rigged_mesh(rig_result, output_path)
            
            return {
                'success': True,
                'rig_path': output_path,
                'skeleton': rig_result['skeleton'],
                'weights': rig_result['weights'],
                'joint_names': rig_result['joint_names']
            }
            
        except Exception as e:
            logger.error(f"Rigging failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _run_rigging_process(self, mesh: trimesh.Trimesh, 
                           rig_type: str) -> Dict[str, Any]:
        """Run the actual rigging process"""
        # This is a simplified version of UniRig processing
        # In production, you would use the actual UniRig pipeline
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Generate humanoid skeleton
        skeleton = self._generate_humanoid_skeleton(vertices)
        
        # Calculate skinning weights
        weights = self._calculate_skinning_weights(vertices, skeleton)
        
        # Get joint names
        joint_names = self._get_joint_names(rig_type)
        
        return {
            'skeleton': skeleton,
            'weights': weights,
            'joint_names': joint_names,
            'vertices': vertices,
            'faces': faces
        }
    
    def _generate_humanoid_skeleton(self, vertices: np.ndarray) -> Dict[str, Any]:
        """Generate a basic humanoid skeleton"""
        # Calculate bounding box
        min_vals = np.min(vertices, axis=0)
        max_vals = np.max(vertices, axis=0)
        center = (min_vals + max_vals) / 2
        height = max_vals[1] - min_vals[1]
        
        # Define joint positions relative to the mesh
        joint_positions = {
            'root': center,
            'pelvis': center + np.array([0, -height * 0.1, 0]),
            'spine_01': center + np.array([0, height * 0.1, 0]),
            'spine_02': center + np.array([0, height * 0.25, 0]),
            'spine_03': center + np.array([0, height * 0.35, 0]),
            'neck': center + np.array([0, height * 0.42, 0]),
            'head': center + np.array([0, height * 0.48, 0]),
            
            # Left arm
            'clavicle_l': center + np.array([-0.08 * height, height * 0.38, 0]),
            'upperarm_l': center + np.array([-0.15 * height, height * 0.35, 0]),
            'lowerarm_l': center + np.array([-0.25 * height, height * 0.25, 0]),
            'hand_l': center + np.array([-0.32 * height, height * 0.15, 0]),
            
            # Right arm
            'clavicle_r': center + np.array([0.08 * height, height * 0.38, 0]),
            'upperarm_r': center + np.array([0.15 * height, height * 0.35, 0]),
            'lowerarm_r': center + np.array([0.25 * height, height * 0.25, 0]),
            'hand_r': center + np.array([0.32 * height, height * 0.15, 0]),
            
            # Left leg
            'thigh_l': center + np.array([-0.08 * height, -height * 0.15, 0]),
            'calf_l': center + np.array([-0.08 * height, -height * 0.35, 0]),
            'foot_l': center + np.array([-0.08 * height, -height * 0.48, 0.05 * height]),
            
            # Right leg
            'thigh_r': center + np.array([0.08 * height, -height * 0.15, 0]),
            'calf_r': center + np.array([0.08 * height, -height * 0.35, 0]),
            'foot_r': center + np.array([0.08 * height, -height * 0.48, 0.05 * height]),
        }
        
        # Define bone hierarchy
        bone_hierarchy = {
            'root': ['pelvis'],
            'pelvis': ['spine_01', 'thigh_l', 'thigh_r'],
            'spine_01': ['spine_02'],
            'spine_02': ['spine_03'],
            'spine_03': ['neck', 'clavicle_l', 'clavicle_r'],
            'neck': ['head'],
            'head': [],
            'clavicle_l': ['upperarm_l'],
            'upperarm_l': ['lowerarm_l'],
            'lowerarm_l': ['hand_l'],
            'hand_l': [],
            'clavicle_r': ['upperarm_r'],
            'upperarm_r': ['lowerarm_r'],
            'lowerarm_r': ['hand_r'],
            'hand_r': [],
            'thigh_l': ['calf_l'],
            'calf_l': ['foot_l'],
            'foot_l': [],
            'thigh_r': ['calf_r'],
            'calf_r': ['foot_r'],
            'foot_r': [],
        }
        
        return {
            'joint_positions': joint_positions,
            'bone_hierarchy': bone_hierarchy
        }
    
    def _calculate_skinning_weights(self, vertices: np.ndarray, 
                                   skeleton: Dict[str, Any]) -> np.ndarray:
        """Calculate skinning weights for each vertex"""
        joint_positions = skeleton['joint_positions']
        joint_names = list(joint_positions.keys())
        num_vertices = len(vertices)
        num_joints = len(joint_names)
        
        # Initialize weights matrix
        weights = np.zeros((num_vertices, num_joints))
        
        # For each vertex, calculate distance to each joint
        for v_idx, vertex in enumerate(vertices):
            distances = []
            for joint_name in joint_names:
                joint_pos = joint_positions[joint_name]
                dist = np.linalg.norm(vertex - joint_pos)
                distances.append(1.0 / (dist + 1e-6))  # Inverse distance
            
            # Normalize weights
            distances = np.array(distances)
            distances = distances / np.sum(distances)
            
            # Apply weight smoothing (limit influence to top 4 joints)
            top_indices = np.argsort(distances)[-4:]
            vertex_weights = np.zeros(num_joints)
            vertex_weights[top_indices] = distances[top_indices]
            vertex_weights = vertex_weights / np.sum(vertex_weights)
            
            weights[v_idx] = vertex_weights
        
        return weights
    
    def _get_joint_names(self, rig_type: str) -> List[str]:
        """Get joint names for specified rig type"""
        if rig_type == 'humanoid':
            return [
                'root', 'pelvis', 'spine_01', 'spine_02', 'spine_03', 
                'neck', 'head',
                'clavicle_l', 'upperarm_l', 'lowerarm_l', 'hand_l',
                'clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r',
                'thigh_l', 'calf_l', 'foot_l',
                'thigh_r', 'calf_r', 'foot_r'
            ]
        else:
            raise ValueError(f"Unsupported rig type: {rig_type}")
    
    def _save_rigged_mesh(self, rig_result: Dict[str, Any], output_path: str):
        """Save rigged mesh with skeleton and weights"""
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save mesh
        mesh = trimesh.Trimesh(
            vertices=rig_result['vertices'],
            faces=rig_result['faces']
        )
        mesh.export(output_path + '.obj')
        
        # Save skeleton data
        skeleton_data = {
            'skeleton': rig_result['skeleton'],
            'joint_names': rig_result['joint_names']
        }
        
        with open(output_path + '_skeleton.json', 'w') as f:
            json.dump(skeleton_data, f, indent=2, default=self._json_serializer)
        
        # Save weights
        np.save(output_path + '_weights.npy', rig_result['weights'])
        
        logger.info(f"Saved rigged mesh to {output_path}")
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy arrays"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)


class RiggingPipeline:
    """Complete rigging pipeline"""
    
    def __init__(self, cache_dir: str = './rigging_cache'):
        self.unirig = UniRigWrapper()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.rigged_models = {}  # Cache for rigged models
    
    def process_reconstruction(self, reconstruction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a reconstructed mesh through rigging
        
        Args:
            reconstruction_result: Result from PIFuHD reconstruction
            
        Returns:
            Rigging results
        """
        if not reconstruction_result.get('success', False):
            return {
                'success': False,
                'error': 'Reconstruction failed'
            }
        
        person_id = reconstruction_result['person_id']
        mesh_path = reconstruction_result['mesh_path']
        
        # Check cache first
        if person_id in self.rigged_models:
            logger.info(f"Using cached rig for person {person_id}")
            return self.rigged_models[person_id]
        
        # Create output path
        output_path = os.path.join(self.cache_dir, f"rigged_person_{person_id}")
        
        # Run rigging
        rig_result = self.unirig.rig_mesh(mesh_path, output_path)
        
        # Add metadata
        rig_result['person_id'] = person_id
        rig_result['timestamp'] = reconstruction_result.get('timestamp')
        
        # Cache result
        if rig_result.get('success', False):
            self.rigged_models[person_id] = rig_result
        
        return rig_result
    
    def get_rigged_model(self, person_id: int) -> Optional[Dict[str, Any]]:
        """Get rigged model from cache"""
        return self.rigged_models.get(person_id)
    
    def clear_cache(self):
        """Clear rigged model cache"""
        self.rigged_models.clear()
    
    def export_for_godot(self, rig_result: Dict[str, Any], 
                        export_path: str) -> Dict[str, Any]:
        """
        Export rigged model in Godot-compatible format
        
        Args:
            rig_result: Rigging result
            export_path: Where to export the model
            
        Returns:
            Export information
        """
        if not rig_result.get('success', False):
            return {
                'success': False,
                'error': 'Rigging failed'
            }
        
        try:
            # Create Godot scene structure
            godot_data = {
                'mesh_path': rig_result['rig_path'] + '.obj',
                'skeleton': rig_result['skeleton'],
                'joint_names': rig_result['joint_names'],
                'animation_ready': True
            }
            
            # Save Godot-compatible data
            with open(export_path + '.json', 'w') as f:
                json.dump(godot_data, f, indent=2)
            
            return {
                'success': True,
                'godot_data_path': export_path + '.json',
                'person_id': rig_result['person_id']
            }
            
        except Exception as e:
            logger.error(f"Godot export failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }