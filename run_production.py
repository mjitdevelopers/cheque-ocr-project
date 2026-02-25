#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production Runner for Cheque OCR Processing
Processes images in batches with GPU support
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import glob
import pandas as pd

from parallel_processor import ParallelProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_image_files(input_dir):
    """Get all image files from input directory"""
    input_path = Path(input_dir)
    extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(ext))
    
    return sorted(image_files)

def process_in_batches(file_list, batch_size, processor_args):
    """Process files in batches and combine results"""
    
    all_results = []
    total_files = len(file_list)
    successful_total = 0
    failed_total = 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📦 Starting batch processing of {total_files} files")
    logger.info(f"📦 Batch size: {batch_size}")
    logger.info(f"{'='*60}")
    
    for i in range(0, total_files, batch_size):
        batch_num = i//batch_size + 1
        batch_files = file_list[i:i+batch_size]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 Processing Batch {batch_num}/{((total_files-1)//batch_size)+1}: {len(batch_files)} files")
        logger.info(f"{'='*60}")
        
        # Initialize processor for this batch
        processor = ParallelProcessor(**processor_args)
        
        # Process batch
        try:
            stats = processor.run(batch_files)
            
            # Check if stats is not None before accessing
            if stats:
                successful = stats.get('successful', 0)
                failed = stats.get('failed', 0)
                logger.info(f"✅ Batch {batch_num} complete: {successful} successful, {failed} failed")
                successful_total += successful
                failed_total += failed
            else:
                logger.warning(f"⚠️ Batch {batch_num} returned no stats")
                successful_total += 0
                failed_total += len(batch_files)
                
        except Exception as e:
            logger.error(f"❌ Batch {batch_num} failed: {str(e)}")
            successful_total += 0
            failed_total += len(batch_files)
            import traceback
            traceback.print_exc()
    
    # Create combined summary file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_dir = Path(processor_args['output_dir'])
    combined_file = combined_dir / f"COMBINED_SUMMARY_{timestamp}.csv"
    
    # Combine all CSV files
    csv_files = []
    for pattern in ['PAYEE_NAMES_*.csv', 'ISSUER_NAMES_*.csv']:
        csv_files.extend(glob.glob(str(combined_dir / "batch_*" / pattern)))
    
    if csv_files:
        logger.info(f"\n📊 Combining {len(csv_files)} CSV files...")
        
        # Group by type
        payee_files = [f for f in csv_files if 'PAYEE_NAMES' in f]
        issuer_files = [f for f in csv_files if 'ISSUER_NAMES' in f]
        
        # Combine payee files
        if payee_files:
            payee_dfs = []
            for f in payee_files:
                try:
                    df = pd.read_csv(f)
                    payee_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Could not read {f}: {e}")
            
            if payee_dfs:
                combined_payee = pd.concat(payee_dfs, ignore_index=True)
                payee_summary = combined_dir / f"ALL_PAYEE_NAMES_{timestamp}.csv"
                combined_payee.to_csv(payee_summary, index=False)
                logger.info(f"✅ Combined payee file: {payee_summary}")
        
        # Combine issuer files
        if issuer_files:
            issuer_dfs = []
            for f in issuer_files:
                try:
                    df = pd.read_csv(f)
                    issuer_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Could not read {f}: {e}")
            
            if issuer_dfs:
                combined_issuer = pd.concat(issuer_dfs, ignore_index=True)
                issuer_summary = combined_dir / f"ALL_ISSUER_NAMES_{timestamp}.csv"
                combined_issuer.to_csv(issuer_summary, index=False)
                logger.info(f"✅ Combined issuer file: {issuer_summary}")
        
        # Create a simple text summary
        summary_txt = combined_dir / f"PROCESSING_SUMMARY_{timestamp}.txt"
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CHEQUE OCR PROCESSING SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total files processed: {total_files}\n")
            f.write(f"Successfully processed: {successful_total}\n")
            f.write(f"Failed: {failed_total}\n")
            f.write(f"Success rate: {(successful_total/total_files*100):.1f}%\n\n")
            f.write("Output files:\n")
            if 'payee_summary' in locals():
                f.write(f"  - Payee names: {payee_summary.name}\n")
            if 'issuer_summary' in locals():
                f.write(f"  - Issuer names: {issuer_summary.name}\n")
        
        logger.info(f"✅ Summary file: {summary_txt}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Total files: {total_files}")
    logger.info(f"✅ Successful: {successful_total}")
    logger.info(f"❌ Failed: {failed_total}")
    logger.info(f"📈 Success rate: {(successful_total/total_files*100):.1f}%")
    logger.info(f"{'='*60}")
    
    return combined_file if 'combined_file' in locals() else None

def main():
    parser = argparse.ArgumentParser(description='Production Cheque OCR Processing')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory with cheque images')
    parser.add_argument('--output-dir', '-o', default='./results', help='Output directory (default: ./results)')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Batch size (default: 100)')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of files to process')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for OCR')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--gpu-mem', type=int, default=4000, help='GPU memory in MB (default: 4000)')
    
    args = parser.parse_args()
    
    # Get all image files
    all_files = get_image_files(args.input_dir)
    
    if not all_files:
        logger.error(f"No image files found in {args.input_dir}")
        sys.exit(1)
    
    # Apply limit if specified
    if args.limit:
        all_files = all_files[:args.limit]
    
    logger.info(f"📸 Found {len(all_files)} image files")
    
    # Processor arguments
    processor_args = {
        'num_cpus': 1,
        'processes_per_cpu': 2,
        'cpu_threads_per_process': 1,
        'output_dir': args.output_dir,
        'use_gpu': args.use_gpu,
        'gpu_id': args.gpu_id,
        'gpu_mem': args.gpu_mem
    }
    
    # Process in batches
    combined_file = process_in_batches(all_files, args.batch_size, processor_args)
    
    logger.info("\n✨ All done!")

if __name__ == "__main__":
    main()