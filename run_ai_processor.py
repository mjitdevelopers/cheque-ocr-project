#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production runner for AI-Powered Cheque Processing
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import torch

from ai_processor import AIChequeProcessor, AIWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-Powered Cheque Processing Production Runner')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory with cheque images')
    parser.add_argument('--output-dir', '-o', default='./ai_results', help='Output directory')
    parser.add_argument('--batch-size', '-b', type=int, default=50, help='Batch size')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of files')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.use_gpu:
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available with {torch.cuda.device_count()} GPUs")
            logger.info(f"   Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
        else:
            logger.warning("⚠️ GPU requested but CUDA not available, falling back to CPU")
            args.use_gpu = False
    
    # Get image files
    input_path = Path(args.input_dir)
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(input_path.glob(ext))
    
    image_files = sorted(image_files)
    
    if not image_files:
        logger.error(f"No image files found in {args.input_dir}")
        sys.exit(1)
    
    if args.limit:
        image_files = image_files[:args.limit]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🤖 AI-POWERED CHEQUE PROCESSING SYSTEM")
    logger.info(f"{'='*60}")
    logger.info(f"📸 Found {len(image_files)} images")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"🎮 GPU Mode: {'Enabled' if args.use_gpu else 'Disabled'}")
    logger.info(f"{'='*60}\n")
    
    # Initialize AI processor
    processor = AIChequeProcessor(
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id
    )
    
    # Initialize writer
    writer = AIWriter(args.output_dir)
    
    # Process in batches
    for i in range(0, len(image_files), args.batch_size):
        batch_num = i // args.batch_size + 1
        batch_files = image_files[i:i + args.batch_size]
        
        logger.info(f"\n📦 Processing Batch {batch_num} ({len(batch_files)} files)")
        
        for j, image_path in enumerate(batch_files, 1):
            logger.info(f"   [{j}/{len(batch_files)}] {image_path.name}")
            
            result = processor.process_cheque(str(image_path))
            writer.write_result(result)
    
    # Save detailed results
    writer.save_detailed_json()
    
    logger.info("\n" + "="*60)
    logger.info("🎉 AI PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"📊 Results:")
    logger.info(f"   - CSV: {writer.csv_path}")
    logger.info(f"   - JSON: {writer.json_path}")
    logger.info(f"   - Summary: {writer.txt_path}")
    logger.info("="*60)

if __name__ == "__main__":
    main()