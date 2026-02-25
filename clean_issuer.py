#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
REAL AI ISSUER EXTRACTOR - Directly from Cheque Images
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import csv
import re
from paddleocr import PaddleOCR
from transformers import pipeline

class RealAIExtractor:
    """Extract issuer names DIRECTLY from cheque images"""
    
    def __init__(self):
        print("="*60)
        print("🤖 REAL ISSUER EXTRACTOR - FROM CHEQUE IMAGES")
        print("="*60)
        
        # Load OCR
        print("\n📥 Loading OCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("✅ OCR loaded")
        
        # Load AI model for name extraction
        print("\n📥 Loading AI Name Extractor...")
        self.ner = pipeline("ner", model="dslim/bert-base-NER", device=-1)
        print("✅ AI Model loaded")
        
        self.results = []
    
    def extract_issuer_from_image(self, image_path):
        """Extract issuer name directly from image"""
        
        # 1. OCR se text nikaalo
        result = self.ocr.ocr(str(image_path), cls=True)
        
        # 2. Text ko combine karo
        text = ""
        if result and result[0]:
            for line in result[0]:
                text += line[1][0] + " "
        
        if not text:
            return "NO_TEXT_FOUND"
        
        print(f"\n📝 OCR Text: {text[:200]}...")
        
        # 3. AI se issuer name dhundo
        entities = self.ner(text[:512])
        
        # 4. Issuer name extract karo (usually after "FOR" or before "AUTHORISED")
        issuer_candidates = []
        for entity in entities:
            if entity['entity'] in ['B-PER', 'I-PER', 'B-ORG', 'I-ORG']:
                word = entity['word']
                score = entity['score']
                
                # Check if this might be issuer (context)
                start = max(0, entity['start'] - 30)
                context = text[start:entity['start']].upper()
                
                if 'FOR' in context or 'AUTHORISED' in context or 'SIGN' in context:
                    issuer_candidates.append((word, score))
        
        # 5. Best candidate lo
        if issuer_candidates:
            best = max(issuer_candidates, key=lambda x: x[1])
            return best[0].strip()
        
        # 6. Agar koi candidate na mile, last line try karo
        lines = text.split('\n')
        for line in reversed(lines[-5:]):
            line = line.strip()
            if line and len(line) > 5 and line.isupper():
                return line
        
        return "ISSUER_NOT_FOUND"
    
    def process_folder(self, folder_path, limit=None):
        """Process all images in folder"""
        folder = Path(folder_path)
        
        # Get all images
        images = []
        for ext in ['*.tif', '*.tiff', '*.jpg', '*.png']:
            images.extend(folder.glob(ext))
        
        images = sorted(images)
        if limit:
            images = images[:limit]
        
        print(f"\n📸 Found {len(images)} cheque images")
        
        # CSV file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = folder / f"ISSUER_NAMES_{timestamp}.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Issuer_Name', 'Confidence'])
        
        # Process each image
        for i, img in enumerate(images, 1):
            print(f"\n{'='*50}")
            print(f"[{i}/{len(images)}] Processing: {img.name}")
            print(f"{'='*50}")
            
            issuer = self.extract_issuer_from_image(img)
            
            # Confidence approximate
            confidence = 0.8 if issuer not in ['NO_TEXT_FOUND', 'ISSUER_NOT_FOUND'] else 0.1
            
            # Save
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([img.name, issuer, confidence])
            
            print(f"✅ Issuer: {issuer}")
        
        print(f"\n🎉 Processing complete!")
        print(f"📁 Results saved: {csv_file}")
        return csv_file

# Run
if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "F_23022026_010"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    extractor = RealAIExtractor()
    extractor.process_folder(folder, limit)