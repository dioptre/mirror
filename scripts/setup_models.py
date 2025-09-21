#!/usr/bin/env python3
"""
Model Setup Script for Avatar Mirror System
Downloads and sets up all required AI models and repositories
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.model_manager import ModelManager

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='Setup AI models for Avatar Mirror System')
    parser.add_argument('--models-dir', default='./models', help='Directory to store models')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--basic-only', action='store_true', help='Only download basic models')
    parser.add_argument('--repositories-only', action='store_true', help='Only setup repositories')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--storage-info', action='store_true', help='Show storage information')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Avatar Mirror System - Model Setup")
    
    # Initialize model manager
    model_manager = ModelManager(models_dir=args.models_dir)
    
    if args.list_models:
        logger.info("📋 Available Models:")
        models = model_manager.list_available_models()
        for category, models_in_category in models.items():
            print(f"\n📁 {category.upper()}:")
            for model_name, model_info in models_in_category.items():
                if model_info.get('status') == 'auto':
                    print(f"  🤖 {model_name}: {model_info.get('description', 'Auto-downloaded')}")
                elif 'files' in model_info:
                    print(f"  📦 {model_name}:")
                    for file_info in model_info['files']:
                        status = "✅" if file_info['exists'] else "❌"
                        optional = " (optional)" if file_info['optional'] else ""
                        print(f"    {status} {file_info['name']} ({file_info['size']}){optional}")
        return
    
    if args.storage_info:
        info = model_manager.get_storage_info()
        logger.info("💾 Storage Information:")
        logger.info(f"  Models directory: {info['models_dir']}")
        logger.info(f"  Total size: {info['total_size_mb']:.1f} MB")
        logger.info(f"  File count: {info['file_count']}")
        return
    
    success = True
    
    if not args.repositories_only:
        # Download basic models
        logger.info("📥 Downloading basic models...")
        results = model_manager.check_and_download_models()
        
        failed_categories = [cat for cat, success in results.items() if not success]
        if failed_categories:
            logger.warning(f"❌ Failed to download models for: {', '.join(failed_categories)}")
            success = False
        else:
            logger.info("✅ All basic models downloaded successfully")
    
    if not args.basic_only:
        # Setup repositories
        logger.info("📦 Setting up AI model repositories...")
        repo_results = model_manager.setup_all_repositories()
        
        failed_repos = [repo for repo, success in repo_results.items() if not success]
        if failed_repos:
            logger.warning(f"❌ Failed to setup repositories: {', '.join(failed_repos)}")
            success = False
        else:
            logger.info("✅ All repositories set up successfully")
    
    # Show final status
    if success:
        logger.info("🎉 Model setup completed successfully!")
        
        # Show storage info
        info = model_manager.get_storage_info()
        logger.info(f"💾 Total storage used: {info['total_size_mb']:.1f} MB ({info['file_count']} files)")
        
        logger.info("\n🚀 Next steps:")
        logger.info("  1. Run the avatar mirror system: make run")
        logger.info("  2. The system will automatically use the downloaded models")
        logger.info("  3. For additional model weights (PIFuHD checkpoints), check the respective repositories")
        
    else:
        logger.error("❌ Model setup completed with errors")
        logger.info("🔧 The system will still work with available models and fallbacks")
        sys.exit(1)

if __name__ == '__main__':
    main()