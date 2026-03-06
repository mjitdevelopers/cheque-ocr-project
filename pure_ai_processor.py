#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PURE AI Cheque Processing - No pattern fallback, only AI models
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
import time

from ocr_engine import OCREngine
from tiff_processor import get_tiff_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PureAIChequeProcessor:
    """100% AI-powered cheque processing - no pattern matching fallbacks"""
    
    def __init__(self, use_gpu=True, gpu_id=0):
        self.device = torch.device(f'cuda:{gpu_id}' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🤖 Initializing PURE AI Processor on {self.device}")
        logger.info("Loading AI models... (this may take a moment)")
        
        # OCR Engine (already AI)
        self.ocr_engine = OCREngine(use_gpu=use_gpu, gpu_id=gpu_id)
        self.tiff_processor = get_tiff_processor(scale_percent=30)
        
        # Load ALL AI models at startup
        self._load_ai_models()
        
        # Performance tracking
        self.processing_times = []
        logger.info("✅ Pure AI Processor ready!")
    
    def _load_ai_models(self):
        """Load all AI models"""
        
        # 1. Main NER model for entity extraction
        logger.info("📥 Loading NER model...")
        self.ner_model = pipeline(
            "ner",
            model="dslim/distilbert-NER",
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="simple"
        )
        logger.info("✅ NER model loaded")
        
        # 2. Zero-shot classifier for context understanding
        logger.info("📥 Loading zero-shot classifier...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("✅ Zero-shot classifier loaded")
        
        # 3. Question-answering model for specific queries
        logger.info("📥 Loading QA model...")
        self.qa_model = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("✅ QA model loaded")
    
    def extract_payee_with_ai(self, text: str) -> Dict:
        """Extract payee name using ONLY AI"""
        
        # Method 1: Use QA model to ask directly
        qa_result = self.qa_model({
            'question': 'Who is the payee or beneficiary of this cheque?',
            'context': text[:512]
        })
        
        if qa_result['score'] > 0.3:
            return {
                'payee': qa_result['answer'].title(),
                'confidence': qa_result['score'],
                'method': 'qa_model'
            }
        
        # Method 2: Use NER to find person/org entities
        ner_results = self.ner_model(text[:512])
        
        if ner_results:
            # Filter for entities that look like payees
            for entity in ner_results:
                # Check context around the entity
                context = text[max(0, entity['start']-50):min(len(text), entity['end']+50)]
                
                # Use zero-shot to verify if this is a payee
                verification = self.classifier(
                    context,
                    candidate_labels=['payee name', 'bank detail', 'amount', 'date']
                )
                
                if verification['labels'][0] == 'payee name' and verification['scores'][0] > 0.6:
                    return {
                        'payee': entity['word'].title(),
                        'confidence': entity['score'] * verification['scores'][0],
                        'method': 'ner_plus_context'
                    }
        
        # Method 3: If all else fails, use zero-shot on entire text
        result = self.classifier(
            text[:512],
            candidate_labels=['payee name', 'issuer name', 'bank name', 'other']
        )
        
        return {
            'payee': 'AI_COULD_NOT_DETERMINE',
            'confidence': result['scores'][0],
            'method': 'zero_shot_fallback'
        }
    
    def extract_issuer_with_ai(self, text: str) -> Dict:
        """Extract issuer name using ONLY AI"""
        
        # Method 1: Use QA model
        qa_result = self.qa_model({
            'question': 'Who is the drawer or issuer of this cheque? Look for names after FOR or before AUTHORISED',
            'context': text[:512]
        })
        
        if qa_result['score'] > 0.3:
            return {
                'issuer': qa_result['answer'].title(),
                'confidence': qa_result['score'],
                'method': 'qa_model'
            }
        
        # Method 2: Use NER with issuer-specific context
        ner_results = self.ner_model(text[:512])
        
        if ner_results:
            for entity in ner_results:
                context = text[max(0, entity['start']-50):min(len(text), entity['end']+50)]
                
                # Look for issuer indicators
                if 'FOR' in context.upper() or 'AUTHORISED' in context.upper():
                    verification = self.classifier(
                        context,
                        candidate_labels=['issuer name', 'signature', 'authorization']
                    )
                    
                    if verification['labels'][0] == 'issuer name':
                        return {
                            'issuer': entity['word'].title(),
                            'confidence': entity['score'] * verification['scores'][0],
                            'method': 'ner_with_context'
                        }
        
        # Method 3: Find the most name-like entity near the bottom
        lines = text.split('\n')
        bottom_text = '\n'.join(lines[-10:])  # Last 10 lines
        
        ner_bottom = self.ner_model(bottom_text[:512])
        
        if ner_bottom:
            best = max(ner_bottom, key=lambda x: x['score'])
            return {
                'issuer': best['word'].title(),
                'confidence': best['score'] * 0.8,  # Slightly lower confidence for bottom
                'method': 'bottom_ner'
            }
        
        return {
            'issuer': 'AI_COULD_NOT_DETERMINE',
            'confidence': 0.1,
            'method': 'failed'
        }
    
    def process_cheque(self, image_path: str) -> Dict:
        """Process a single cheque using ONLY AI"""
        start_time = time.time()
        
        try:
            # OCR (AI-powered)
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
            
            # AI-ONLY extraction (no patterns!)
            payee_result = self.extract_payee_with_ai(text)
            issuer_result = self.extract_issuer_with_ai(text)
            
            # Overall confidence (weighted average)
            overall_confidence = (
                payee_result['confidence'] * 0.4 +
                issuer_result['confidence'] * 0.4 +
                ocr_result.get('confidence', 0) * 0.2
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.processing_times.append(elapsed_ms)
            
            result = {
                'filename': Path(image_path).name,
                'success': True,
                'payee': payee_result['payee'],
                'payee_confidence': round(payee_result['confidence'], 3),
                'payee_method': payee_result['method'],
                'issuer': issuer_result['issuer'],
                'issuer_confidence': round(issuer_result['confidence'], 3),
                'issuer_method': issuer_result['method'],
                'overall_confidence': round(overall_confidence, 3),
                'time_ms': elapsed_ms,
                'ocr_confidence': ocr_result.get('confidence', 0)
            }
            
            # Log result
            logger.info(f"✅ {Path(image_path).name}: Payee={result['payee']} ({result['payee_confidence']}), Issuer={result['issuer']} ({result['issuer_confidence']})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'filename': Path(image_path).name,
                'success': False,
                'error': str(e),
                'time_ms': int((time.time() - start_time) * 1000)
            }


class AIOnlyWriter:
    """Write AI-only results"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.output_dir / f"PURE_AI_RESULTS_{timestamp}.csv"
        self.json_path = self.output_dir / f"PURE_AI_DETAILED_{timestamp}.json"
        
        self.results = []
        
        # Initialize CSV
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Filename', 'Payee', 'Payee_Confidence', 'Payee_Method',
                'Issuer', 'Issuer_Confidence', 'Issuer_Method',
                'Overall_Confidence', 'Time_ms'
            ])
        
        print(f"\n🤖 PURE AI RESULTS: {self.csv_path}\n")
    
    def write_result(self, result):
        """Write result to files"""
        self.results.append(result)
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.get('filename', ''),
                result.get('payee', 'ERROR'),
                result.get('payee_confidence', 0),
                result.get('payee_method', ''),
                result.get('issuer', 'ERROR'),
                result.get('issuer_confidence', 0),
                result.get('issuer_method', ''),
                result.get('overall_confidence', 0),
                result.get('time_ms', 0)
            ])
    
    def save_json(self):
        """Save detailed JSON"""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total': len(self.results),
                'results': self.results
            }, f, indent=2, default=str)
        
        print(f"\n📊 Detailed JSON: {self.json_path}")


# Runner script
if __name__ == "__main__":
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    
    def get_images(input_dir, limit=None):
        path = Path(input_dir)
        images = []
        for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']:
            images.extend(path.glob(ext))
        return sorted(images)[:limit] if limit else sorted(images)
    
    if len(sys.argv) < 2:
        print("Usage: python pure_ai_processor.py <input_dir> [limit]")
        print("Example: python pure_ai_processor.py F_23022026_010 10")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Get images
    images = get_images(input_dir, limit)
    print(f"📸 Found {len(images)} images")
    
    # Initialize processor and writer
    processor = PureAIChequeProcessor(use_gpu=True)
    writer = AIOnlyWriter("./pure_ai_results")
    
    # Process with thread pool
    with ThreadPoolExecutor(max_workers=2) as executor:  # Fewer workers for AI
        futures = {executor.submit(processor.process_cheque, img): img for img in images}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            writer.write_result(result)
            
            if i % 5 == 0:
                print(f"Progress: {i}/{len(images)}")
    
    # Save detailed JSON
    writer.save_json()
    
    # Print statistics
    successful = [r for r in writer.results if r.get('success')]
    avg_time = np.mean([r.get('time_ms', 0) for r in successful])
    
    print(f"\n{'='*60}")
    print("PURE AI PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(writer.results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(writer.results) - len(successful)}")
    print(f"Average time: {avg_time:.0f}ms")
    print(f"{'='*60}")
    print(f"📁 Results: {writer.csv_path}")