#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIFF Image Processor for Cheque OCR
Optimizes large TIFF files for PaddleOCR performance
Based on proven benchmark: 50% scaling reduces 60s → 15-20s
Last Updated: 2025
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)


class TIFFProcessor:
    """
    Optimize TIFF cheques for OCR processing
    Handles multi-page TIFFs, large formats, and color spaces
    """
    
    def __init__(self, scale_percent: int = 50, target_long_side: int = 1200):
        """
        Initialize TIFF processor
        
        Args:
            scale_percent: Percentage to scale (50 = half size)
            target_long_side: Alternative - resize to specific long side length
        """
        self.scale_percent = scale_percent
        self.target_long_side = target_long_side
        
        # Performance tracking
        self.stats = {
            'processed': 0,
            'total_time': 0,
            'avg_time': 0,
            'original_sizes': [],
            'processed_sizes': []
        }
    
    def preprocess(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
        """
        Preprocess TIFF for optimal OCR performance
        Returns: (processed image, metadata)
        """
        start_time = cv2.getTickCount()
        image_path = Path(image_path)
        
        metadata = {
            'original_path': str(image_path),
            'original_size_mb': 0,
            'processed_size_mb': 0,
            'scale_factor': 1.0,
            'processing_time_ms': 0
        }
        
        try:
            # Read TIFF (supports multi-page, but we take first page)
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            # Get original size
            original_height, original_width = img.shape[:2]
            metadata['original_dimensions'] = f"{original_width}x{original_height}"
            metadata['original_size_mb'] = os.path.getsize(image_path) / (1024 * 1024)
            
            # Convert to BGR if needed
            if len(img.shape) == 2:
                # Grayscale to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                # RGBA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Calculate scale factor
            if self.target_long_side:
                # Scale based on target long side
                long_side = max(original_width, original_height)
                scale_factor = self.target_long_side / long_side
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
            else:
                # Scale by percentage
                scale_factor = self.scale_percent / 100
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
            
            metadata['scale_factor'] = scale_factor
            
            # Resize
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Optional: Enhance contrast for better OCR
            if np.mean(img) < 128:
                # Dark image - enhance
                img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
            
            # Calculate processed size (estimate)
            metadata['processed_dimensions'] = f"{new_width}x{new_height}"
            metadata['processed_size_mb'] = (new_width * new_height * 3) / (1024 * 1024)
            
            # Update stats
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
            metadata['processing_time_ms'] = processing_time
            
            self.stats['processed'] += 1
            self.stats['total_time'] += processing_time
            self.stats['avg_time'] = self.stats['total_time'] / self.stats['processed']
            self.stats['original_sizes'].append(metadata['original_size_mb'])
            self.stats['processed_sizes'].append(metadata['processed_size_mb'])
            
            logger.debug(f"Processed {image_path.name}: {original_width}x{original_height} → {new_width}x{new_height} "
                        f"({scale_factor:.2f}x) in {processing_time:.1f}ms")
            
            return img, metadata
            
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            # Return original if processing fails
            img = cv2.imread(str(image_path))
            return img, metadata
    
    def batch_preprocess(self, image_paths: list, max_workers: int = 4) -> list:
        """
        Preprocess multiple TIFFs in parallel
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.preprocess, path) for path in image_paths]
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch preprocessing failed: {e}")
                    results.append((None, {'error': str(e)}))
        
        return results
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        if self.stats['processed'] == 0:
            return self.stats
        
        stats = self.stats.copy()
        stats['avg_original_mb'] = np.mean(self.stats['original_sizes']) if self.stats['original_sizes'] else 0
        stats['avg_processed_mb'] = np.mean(self.stats['processed_sizes']) if self.stats['processed_sizes'] else 0
        stats['compression_ratio'] = (stats['avg_processed_mb'] / stats['avg_original_mb'] 
                                      if stats['avg_original_mb'] > 0 else 1)
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'processed': 0,
            'total_time': 0,
            'avg_time': 0,
            'original_sizes': [],
            'processed_sizes': []
        }


# Singleton instance
_tiff_processor_instance = None

def get_tiff_processor(scale_percent: int = 50) -> TIFFProcessor:
    """Get or create TIFF processor singleton"""
    global _tiff_processor_instance
    if _tiff_processor_instance is None:
        _tiff_processor_instance = TIFFProcessor(scale_percent=scale_percent)
    return _tiff_processor_instance