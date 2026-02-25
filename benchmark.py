#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULTRA-SIMPLE BENCHMARK - Guaranteed to work
"""

import time
import json
import platform
import psutil
from pathlib import Path
import logging
import cv2
import numpy as np

from ocr_engine import OCREngine
from tiff_processor import get_tiff_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Simple benchmark"""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <test_image>")
        sys.exit(1)
    
    test_image = sys.argv[1]
    
    print("=" * 50)
    print("🔬 ULTRA-SIMPLE BENCHMARK")
    print("=" * 50)
    
    # System info
    print(f"CPU: {platform.processor()}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"RAM: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    
    # Load image
    tiff_processor = get_tiff_processor(scale_percent=50)
    img, metadata = tiff_processor.preprocess(test_image)
    print(f"Image: {test_image}")
    print(f"Size: {metadata['original_dimensions']} → {metadata['processed_dimensions']}")
    
    # Test with default settings
    print("\n📊 Testing default configuration...")
    
    try:
        # Create engine
        engine = OCREngine(cpu_threads=2, enable_mkldnn=True)
        
        # Warmup
        print("  Warming up...")
        _ = engine.process_cheque(img)
        
        # Test 3 runs
        times = []
        for i in range(3):
            start = time.time()
            result = engine.process_cheque(img)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s - Payee: {result['payee_cleaned']}")
        
        avg_time = sum(times) / len(times)
        images_per_min = 60 / avg_time
        
        print(f"\n✅ Results:")
        print(f"  Average time: {avg_time:.2f} seconds")
        print(f"  Speed: {images_per_min:.1f} images/minute")
        print(f"  Success: {result['success']}")
        
        # Projections
        print(f"\n📈 Projections:")
        print(f"  1 process: {images_per_min:.0f} images/min")
        print(f"  2 processes: {images_per_min*2*0.85:.0f} images/min")
        print(f"  50,000 images: {50000/(images_per_min*2*0.85)/60:.1f} hours")
        
        # Save results
        results = {
            'system': {
                'cpu': platform.processor(),
                'cores': psutil.cpu_count(logical=False),
                'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            },
            'avg_time_seconds': round(avg_time, 2),
            'images_per_minute': round(images_per_min, 1),
            'sample_result': {
                'payee': result['payee_cleaned'],
                'confidence': result['confidence'],
                'is_government': result['is_government']
            }
        }
        
        with open('benchmark_result.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to benchmark_result.json")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()