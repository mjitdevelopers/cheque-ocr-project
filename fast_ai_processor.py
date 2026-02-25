#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast AI-Powered Cheque Processing - NO BROKEN DEPENDENCIES
Uses transformers only, no spellchecker or flair
"""

import logging
from pathlib import Path
from datetime import datetime
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import csv
import json
import re
from typing import Dict, List, Optional
import numpy as np

from ocr_engine import OCREngine
from tiff_processor import get_tiff_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastAIChequeProcessor:
    """Optimized AI processor with no broken dependencies"""
    
    def __init__(self, use_gpu=True, gpu_id=0):
        self.device = torch.device(f'cuda:{gpu_id}' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🚀 Initializing FAST AI Processor on {self.device}")
        
        # OCR Engine
        self.ocr_engine = OCREngine(use_gpu=use_gpu, gpu_id=gpu_id)
        self.tiff_processor = get_tiff_processor(scale_percent=30)  # Reduced for speed
        
        # Use a single small model for NER
        try:
            self.ner_model = pipeline(
                "ner",
                model="dslim/distilbert-NER",  # Smaller, faster model
                device=0 if torch.cuda.is_available() else -1,
                aggregation_strategy="simple"
            )
            logger.info("✅ Fast NER model loaded")
        except Exception as e:
            logger.warning(f"Could not load NER model: {e}")
            self.ner_model = None
        
        # Simple spelling corrections (hardcoded, no extra package)
        self.spelling_corrections = {
            'STEELNDUSTRIES': 'STEEL INDUSTRIES',
            'STEELHDUSTRIES': 'STEEL INDUSTRIES',
            'STEELINDUSTRIES': 'STEEL INDUSTRIES',
            'ENTENPES': 'ENTERPRISES',
            'ENTSEPS': 'ENTERPRISES',
            'ENTERPRISESS': 'ENTERPRISES',
            'CURUGRAM': 'GURUGRAM',
            'GURGAON': 'GURUGRAM',
            'APRLIANGES': 'APPLIANCES',
            'DEVAPRLIANGES': 'DEV APPLIANCES',
            'FASTNERS': 'FASTENERS',
            'KISPA': 'KIXPA',
            'ROHIN': 'ROHIT',
            'POLYPLAST': 'POLYPLAST PVT LTD',
        }
        
        # Cache for results
        self.cache = {}
        
        # Performance tracking
        self.processing_times = []
    
    def extract_with_ai(self, text: str) -> Dict:
        """Extract entities using AI model"""
        if not self.ner_model or not text:
            return {'payee': None, 'issuer': None, 'confidence': 0}
        
        try:
            # Limit text for speed
            text_short = text[:500]
            
            # Get NER predictions
            results = self.ner_model(text_short)
            
            if not results:
                return {'payee': None, 'issuer': None, 'confidence': 0}
            
            # Find best entity
            best = max(results, key=lambda x: x['score'])
            
            # Determine if it's payee or issuer based on surrounding text
            context = text_short[max(0, best['start']-30):best['end']+30].upper()
            
            entity_type = 'issuer' if 'FOR' in context or 'AUTHORISED' in context else 'payee'
            
            return {
                entity_type: best['word'],
                'confidence': best['score'],
                'raw_text': best['word']
            }
            
        except Exception as e:
            logger.debug(f"AI extraction failed: {e}")
            return {'payee': None, 'issuer': None, 'confidence': 0}
    
    def extract_with_patterns(self, text: str) -> Dict:
        """Fast pattern-based extraction"""
        text_upper = text.upper()
        
        result = {
            'payee': None,
            'issuer': None,
            'confidence': 0.6  # Default confidence for patterns
        }
        
        # Payee patterns
        payee_patterns = [
            r'PAY\s+([A-Z][A-Z\s\.]{3,50}?)(?:\s+OR|\s+$|\n)',
            r'ORDER\s+OF\s+([A-Z][A-Z\s\.]{3,50}?)(?:\s+OR|\s+$|\n)',
            r'BENEFICIARY\s+([A-Z][A-Z\s\.]{3,50}?)(?:\s+$|\n)',
        ]
        
        for pattern in payee_patterns:
            match = re.search(pattern, text_upper)
            if match:
                result['payee'] = self.clean_name(match.group(1))
                break
        
        # Issuer patterns
        issuer_patterns = [
            r'FOR\s+([A-Z][A-Z\s\.]{3,50}?)(?:\s+AUTH|\s+SIGN|\s+$)',
            r'([A-Z][A-Z\s\.]{5,50}?)\s+AUTHORISED\s+SIGNATORY',
            r'([A-Z][A-Z\s\.]{5,50}?)\s+SIGNATURE',
        ]
        
        for pattern in issuer_patterns:
            match = re.search(pattern, text_upper)
            if match:
                result['issuer'] = self.clean_name(match.group(1))
                break
        
        return result
    
    def clean_name(self, name: str) -> str:
        """Clean and correct name spelling"""
        if not name:
            return "UNKNOWN"
        
        # Apply corrections
        name_upper = name.upper()
        for wrong, correct in self.spelling_corrections.items():
            if wrong in name_upper:
                name_upper = name_upper.replace(wrong, correct)
        
        # Remove special chars and extra spaces
        name_upper = re.sub(r'[^\w\s]', ' ', name_upper)
        name_upper = ' '.join(name_upper.split())
        
        return name_upper.title() if len(name_upper) >= 3 else "UNKNOWN"
    
    def process_cheque(self, image_path: str) -> Dict:
        """Process a single cheque image"""
        import time
        start_time = time.time()
        
        try:
            # OCR
            img, metadata = self.tiff_processor.preprocess(image_path)
            ocr_result = self.ocr_engine.process_cheque(img, str(image_path))
            
            if not ocr_result.get('success'):
                return {
                    'filename': Path(image_path).name,
                    'success': False,
                    'error': ocr_result.get('error', 'OCR failed'),
                    'time_ms': int((time.time() - start_time) * 1000)
                }
            
            text = ocr_result.get('full_text', '')
            
            # Try AI first (if available)
            ai_result = self.extract_with_ai(text)
            
            # Fallback to patterns
            pattern_result = self.extract_with_patterns(text)
            
            # Combine results (prefer AI if confident)
            payee = ai_result.get('payee') or pattern_result.get('payee') or "UNKNOWN"
            issuer = ai_result.get('issuer') or pattern_result.get('issuer') or "UNKNOWN"
            
            # Calculate confidence
            if ai_result.get('confidence', 0) > 0.7:
                confidence = ai_result['confidence']
                method = 'ai'
            else:
                confidence = pattern_result.get('confidence', 0.5)
                method = 'patterns'
            
            # Clean names
            payee = self.clean_name(payee) if payee != "UNKNOWN" else "UNKNOWN"
            issuer = self.clean_name(issuer) if issuer != "UNKNOWN" else "UNKNOWN"
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.processing_times.append(elapsed_ms)
            
            result = {
                'filename': Path(image_path).name,
                'success': True,
                'payee': payee,
                'issuer': issuer,
                'confidence': round(confidence, 3),
                'method': method,
                'time_ms': elapsed_ms,
                'ocr_confidence': ocr_result.get('confidence', 0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'filename': Path(image_path).name,
                'success': False,
                'error': str(e),
                'time_ms': int((time.time() - start_time) * 1000)
            }


class FastAIWriter:
    """Fast result writer"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.output_dir / f"FAST_RESULTS_{timestamp}.csv"
        self.stats_path = self.output_dir / f"STATS_{timestamp}.txt"
        
        self.results = []
        
        # Initialize CSV
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Filename', 'Payee', 'Issuer', 'Confidence', 
                'Method', 'Time_ms', 'OCR_Confidence'
            ])
        
        print(f"\n⚡ FAST RESULTS: {self.csv_path}\n")
    
    def write_result(self, result):
        """Write single result"""
        self.results.append(result)
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.get('filename', ''),
                result.get('payee', 'ERROR'),
                result.get('issuer', 'ERROR'),
                result.get('confidence', 0),
                result.get('method', 'unknown'),
                result.get('time_ms', 0),
                result.get('ocr_confidence', 0)
            ])
    
    def write_stats(self):
        """Write statistics"""
        if not self.results:
            return
        
        successful = [r for r in self.results if r.get('success')]
        times = [r.get('time_ms', 0) for r in successful]
        
        with open(self.stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("FAST AI PROCESSING STATISTICS\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total files: {len(self.results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(self.results) - len(successful)}\n\n")
            
            if times:
                f.write(f"Average time: {np.mean(times):.0f}ms\n")
                f.write(f"Median time: {np.median(times):.0f}ms\n")
                f.write(f"Min time: {np.min(times):.0f}ms\n")
                f.write(f"Max time: {np.max(times):.0f}ms\n")
                f.write(f"Total time: {np.sum(times)/1000:.1f}s\n")
        
        print(f"\n📊 Statistics saved to: {self.stats_path}")