"""
PyTorch compatibility fixes for newer versions
"""

import torch
import argparse
import logging

logger = logging.getLogger(__name__)


def fix_pytorch_load_compatibility():
    """
    Fix PyTorch 2.6+ load compatibility issues with older checkpoints
    """
    try:
        # Add safe globals for older checkpoints
        torch.serialization.add_safe_globals([
            argparse.Namespace,
            # Add other common classes found in older checkpoints
        ])
        logger.info("✅ PyTorch load compatibility fixes applied")
        
    except Exception as e:
        logger.warning(f"Could not apply PyTorch compatibility fixes: {e}")


def patch_torch_load():
    """
    Monkey patch torch.load to use weights_only=False by default for compatibility
    """
    try:
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            # Set weights_only=False if not specified, for compatibility
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        logger.info("✅ torch.load patched for compatibility")
        
    except Exception as e:
        logger.warning(f"Could not patch torch.load: {e}")


# Apply fixes on import
fix_pytorch_load_compatibility()
patch_torch_load()