#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core OCR Engine with GPU Support - FIXED VERSION
"""

from paddleocr import PaddleOCR
import time
import logging
from typing import Optional, Dict, Any, List
import numpy as np
import os
from pathlib import Path

from cheque_classifier import get_classifier
from government_payee_rules import get_rule_engine
from payee_cleaner import get_cleaner
from tiff_processor import get_tiff_processor

logger = logging.getLogger(__name__)


class OCREngine:
    
    def __init__(self, cpu_threads: int = 2, enable_mkldnn: bool = True, 
                 use_gpu: bool = False, gpu_id: int = 0, gpu_mem: int = 4000):
        
        self.cpu_threads = cpu_threads
        self.enable_mkldnn = enable_mkldnn
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.gpu_mem = gpu_mem
        
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                gpu_id=gpu_id,
                gpu_mem=gpu_mem,
                enable_mkldnn=enable_mkldnn if not use_gpu else False,
                cpu_threads=cpu_threads,
                show_log=False
            )
            
            if use_gpu:
                logger.info(f"✅ GPU initialized (ID: {gpu_id}, Memory: {gpu_mem}MB)")
            else:
                logger.info("✅ CPU mode initialized")
                
        except Exception as e:
            logger.error(f"GPU init failed, falling back to CPU: {e}")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                enable_mkldnn=True,
                cpu_threads=cpu_threads,
                show_log=False
            )
            self.use_gpu = False
         # ADD THESE FOR BETTER ACCURACY
            det_db_thresh=0.2,           # Lower threshold = more text detection
            det_db_box_thresh=0.3,        # Lower = more boxes
            det_db_unclip_ratio=2.0,      # Better for cheque text
            drop_score=0.3,                # Lower confidence threshold
            rec_image_shape='3, 48, 320',  # Better for longer text
             
        self.classifier = get_classifier()
        self.rule_engine = get_rule_engine()
        self.cleaner = get_cleaner()
        self.tiff_processor = get_tiff_processor(scale_percent=50)
        
        self.stats = {
            'processed': 0,
            'total_time': 0,
            'avg_time': 0,
            'successful': 0,
            'failed': 0,
            'government': 0
        }
    
    def _extract_full_text(self, ocr_result) -> str:
        """Extract full text from PaddleOCR result"""
        if not ocr_result or not ocr_result[0]:
            return ""
        
        text_parts = []
        for line in ocr_result[0]:
            if len(line) >= 2:
                text_parts.append(line[1][0])
        
        return ' '.join(text_parts)
    
    def _calculate_confidence(self, ocr_result) -> float:
        if not ocr_result or not ocr_result[0]:
            return 0.0
        
        confidences = []
        for line in ocr_result[0]:
            if len(line) >= 2:
                confidences.append(line[1][1])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def process_cheque(self, image: np.ndarray, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a single cheque image and extract ALL text"""
        start_time = time.perf_counter()
        
        result = {
            'success': False,
            'image_path': str(image_path) if image_path else 'memory',
            'processing_time_ms': 0,
            'cheque_type': None,
            'is_government': False,
            'government_category': None,
            'payee_raw': None,
            'payee_cleaned': None,
            'full_text': '',  # CRITICAL: This will store ALL OCR text
            'confidence': 0.0,
            'error': None,
            'validation_notes': []
        }
        
        try:
            # Run OCR on the image
            ocr_result = self.ocr.ocr(image, cls=True)
            
            if not ocr_result or not ocr_result[0]:
                result['error'] = 'No text detected'
                result['payee_cleaned'] = 'XXX'
                return result
            
            # CRITICAL: Extract ALL text from OCR result
            full_text = self._extract_full_text(ocr_result)
            result['full_text'] = full_text  # Store the full text
            
            # Log the extracted text for debugging
            filename = Path(image_path).name if image_path else "unknown"
            logger.info(f"📄 OCR Text from {filename}: {full_text[:100]}...")
            
            # Classify cheque type
            classification = self.classifier.classify(image, full_text)
            result['cheque_type'] = classification.type.value
            result['is_government'] = classification.is_government
            result['government_category'] = classification.government_category
            result['validation_notes'].extend(classification.validation_notes or [])
            
            # Extract raw payee from the FULL TEXT
            raw_payee = self.cleaner.extract_payee_from_text(full_text)
            result['payee_raw'] = raw_payee
            
            # Apply government rules if needed
            if classification.is_government:
                self.stats['government'] += 1
                final_payee = self.rule_engine.enforce_payee_rule(
                    classification=classification,
                    extracted_payee=raw_payee,
                    ocr_full_text=full_text
                )
            else:
                final_payee = self.cleaner.clean(raw_payee) if raw_payee else 'XXX'
            
            result['payee_cleaned'] = final_payee
            
            # Calculate confidence
            confidence = self._calculate_confidence(ocr_result)
            result['confidence'] = confidence
            
            # Success
            result['success'] = True
            self.stats['successful'] += 1
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            result['error'] = str(e)
            result['payee_cleaned'] = 'XXX'
            self.stats['failed'] += 1
        
        # Update timing
        processing_time = (time.perf_counter() - start_time) * 1000
        result['processing_time_ms'] = processing_time
        
        self.stats['processed'] += 1
        self.stats['total_time'] += processing_time
        if self.stats['processed'] > 0:
            self.stats['avg_time'] = self.stats['total_time'] / self.stats['processed']
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        if stats['processed'] > 0:
            stats['success_rate'] = (stats['successful'] / stats['processed']) * 100
            stats['avg_time_ms'] = stats['avg_time']
        return stats
    
    def reset_stats(self):
        self.stats = {
            'processed': 0,
            'total_time': 0,
            'avg_time': 0,
            'successful': 0,
            'failed': 0,
            'government': 0
        }