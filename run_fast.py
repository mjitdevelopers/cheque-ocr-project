#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast AI Processor Runner
"""

import sys
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from fast_ai_processor import FastAIChequeProcessor, FastAIWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_images(input_dir, limit=None):
    """Get image files"""
    path = Path(input_dir)
    images = []
    for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']:
        images.extend(path.glob(ext))
    
    images = sorted(images)
    if limit:
        images = images[:limit]
    
    return images

def process_single(processor, image_path):
    """Process single image"""
    return processor.process_cheque(str(image_path))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fast AI Cheque Processing')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory')
    parser.add_argument('--output-dir', '-o', default='./fast_results', help='Output directory')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of files')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    
    args = parser.parse_args()
    
    # Get images
    images = get_images(args.input_dir, args.limit)
    logger.info(f"📸 Found {len(images)} images")
    
    # Initialize processor and writer
    processor = FastAIChequeProcessor(use_gpu=not args.no_gpu)
    writer = FastAIWriter(args.output_dir)
    
    # Process with thread pool
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single, processor, img): img for img in images}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            writer.write_result(result)
            
            if completed % 10 == 0:
                logger.info(f"Progress: {completed}/{len(images)}")
    
    # Write statistics
    writer.write_stats()
    
    logger.info("\n✅ Processing complete!")
    logger.info(f"📊 Results: {writer.csv_path}")

if __name__ == "__main__":
    main()