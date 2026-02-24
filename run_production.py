#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Production Script with GPU and 200-image batches
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import glob

from parallel_processor import ParallelProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ocr_production_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_tiff_files(input_dir: str) -> list:
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return []
    
    extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tiff_files = []
    for ext in extensions:
        tiff_files.extend(input_path.glob(ext))
    
    return sorted(tiff_files)


def process_in_batches(image_paths: list, batch_size: int, processor_args: dict):
    """Process images in batches of specified size"""
    
    total = len(image_paths)
    logger.info(f"📦 Processing {total} images in batches of {batch_size}")
    
    all_payee_files = []
    
    for i in range(0, total, batch_size):
        batch_num = i // batch_size + 1
        batch_files = image_paths[i:i+batch_size]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"📦 BATCH {batch_num}/{(total+batch_size-1)//batch_size}")
        logger.info(f"📸 Images {i+1} to {min(i+batch_size, total)}")
        logger.info('='*50)
        
        # Create batch output directory
        batch_output = Path(processor_args['output_dir']) / f"batch_{batch_num}"
        batch_output.mkdir(exist_ok=True)
        
        # Create processor for this batch
        processor = ParallelProcessor(
            num_cpus=processor_args['num_cpus'],
            processes_per_cpu=processor_args['processes_per_cpu'],
            cpu_threads_per_process=processor_args['threads_per_process'],
            output_dir=str(batch_output),
            use_gpu=processor_args['use_gpu'],
            gpu_id=processor_args['gpu_id'],
            gpu_mem=processor_args['gpu_mem']
        )
        
        # Process batch
        stats = processor.run(batch_files)
        
        # Find payee file for this batch
        payee_files = list(batch_output.glob("PAYEE_NAMES_*.csv"))
        if payee_files:
            all_payee_files.extend(payee_files)
        
        logger.info(f"✅ Batch {batch_num} complete: {stats['successful']} successful")
    
    # Combine all payee files
    if all_payee_files:
        combined_path = Path(processor_args['output_dir']) / "ALL_PAYEE_NAMES.csv"
        with open(combined_path, 'w', encoding='utf-8') as outfile:
            # Write header
            with open(all_payee_files[0], 'r', encoding='utf-8') as first:
                outfile.write(first.readline())
            
            # Write all data
            total_payees = 0
            for f in all_payee_files:
                with open(f, 'r', encoding='utf-8') as infile:
                    next(infile)  # Skip header
                    for line in infile:
                        if line.strip():
                            outfile.write(line)
                            total_payees += 1
            
            logger.info(f"✅ Combined {total_payees} payee names into {combined_path}")
        
        return combined_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description='OCR with GPU and batching')
    parser.add_argument('--input-dir', '-i', required=True,
                       help='Directory containing TIFF cheque images')
    parser.add_argument('--output-dir', '-o', default='./results',
                       help='Output directory (default: ./results)')
    parser.add_argument('--num-cpus', type=int, default=1,
                       help='Number of CPUs (default: 1)')
    parser.add_argument('--processes-per-cpu', type=int, default=2,
                       help='Processes per CPU (default: 2)')
    parser.add_argument('--threads-per-process', type=int, default=1,
                       help='Threads per process (default: 1)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU for processing')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--gpu-mem', type=int, default=4000,
                       help='GPU memory in MB (default: 4000)')
    parser.add_argument('--batch-size', type=int, default=200,
                       help='Images per batch (default: 200)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of images to process')
    
    args = parser.parse_args()
    
    # Find TIFF files
    logger.info(f"🔍 Scanning {args.input_dir} for TIFF files...")
    all_files = find_tiff_files(args.input_dir)
    
    if not all_files:
        logger.error("No TIFF files found")
        sys.exit(1)
    
    # Apply limit if specified
    if args.limit:
        all_files = all_files[:args.limit]
    
    logger.info(f"📊 Found {len(all_files)} TIFF images")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Process in batches
    processor_args = {
        'num_cpus': args.num_cpus,
        'processes_per_cpu': args.processes_per_cpu,
        'threads_per_process': args.threads_per_process,
        'output_dir': args.output_dir,
        'use_gpu': args.use_gpu,
        'gpu_id': args.gpu_id,
        'gpu_mem': args.gpu_mem
    }
    
    combined_file = process_in_batches(all_files, args.batch_size, processor_args)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("🎉 ALL BATCHES COMPLETE!")
    logger.info("="*60)
    
    if combined_file and combined_file.exists():
        logger.info(f"📁 Final payee names: {combined_file}")
        # Try to open in Notepad
        try:
            os.system(f'notepad "{combined_file}"')
        except:
            pass
    else:
        logger.info("📁 Check individual batch folders for results")


if __name__ == "__main__":
    main()