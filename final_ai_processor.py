#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FINAL WORKING AI-ONLY CHEQUE PROCESSOR - WITH PROPER PRINTING
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import csv
import warnings
warnings.filterwarnings('ignore')

# SIMPLE PRINTING - No logging issues
print("="*60)
print("🤖 FINAL AI PROCESSOR STARTING...")
print("="*60)

# Imports
try:
    import torch
    from transformers import pipeline
    from paddleocr import PaddleOCR
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install karo: pip install torch transformers paddleocr")
    sys.exit(1)

class FinalAIProcessor:
    """Simple working AI processor"""
    
    def __init__(self):
        print("\n📥 Loading OCR model...")
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
            print("✅ OCR loaded")
        except Exception as e:
            print(f"❌ OCR failed: {e}")
            sys.exit(1)
        
        print("\n📥 Loading AI model (30 seconds)...")
        try:
            self.ner = pipeline(
                "ner",
                model="dslim/distilbert-NER",
                device=-1
            )
            print("✅ AI Model loaded")
        except Exception as e:
            print(f"❌ AI model failed: {e}")
            sys.exit(1)
        
        self.results = []
        print("\n✅ Ready to process!\n")
    
    def process_image(self, image_path):
        """Process single image"""
        try:
            filename = Path(image_path).name
            print(f"\n📄 Processing: {filename}")
            
            # OCR
            result = self.ocr.ocr(str(image_path), cls=True)
            
            # Extract text
            text = ""
            if result and result[0]:
                for line in result[0]:
                    text += line[1][0] + " "
            
            if not text:
                print("❌ No text found")
                return {'filename': filename, 'success': False, 'error': 'No text found'}
            
            print(f"📝 Text: {text[:100]}...")
            
            # AI Extraction
            entities = self.ner(text[:512])
            
            # Find names
            payee = "UNKNOWN"
            issuer = "UNKNOWN"
            confidence = 0
            
            if entities:
                if isinstance(entities, dict):
                    entities = [entities]
                
                for e in entities:
                    if isinstance(e, dict):
                        word = e.get('word', '')
                        score = e.get('score', 0)
                        
                        if word and len(word) > 2:
                            if score > confidence:
                                confidence = score
                            
                            # Simple assignment
                            if issuer == "UNKNOWN":
                                issuer = word
                            elif payee == "UNKNOWN":
                                payee = word
            
            result_data = {
                'filename': filename,
                'success': True,
                'payee': payee.title(),
                'issuer': issuer.title(),
                'confidence': round(confidence, 3)
            }
            
            print(f"✅ Payee: {result_data['payee']}")
            print(f"✅ Issuer: {result_data['issuer']}")
            return result_data
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'filename': Path(image_path).name, 'success': False, 'error': str(e)}
    
    def process_folder(self, folder_path, limit=None):
        """Process all images"""
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder nahi mila: {folder_path}")
            return
        
        # Get images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            images.extend(folder.glob(ext))
        
        images = sorted(images)
        if limit:
            images = images[:limit]
        
        print(f"\n📸 Total images: {len(images)}")
        
        # CSV file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = folder / f"AI_RESULTS_{timestamp}.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Payee', 'Issuer', 'Confidence'])
        
        # Process images
        successful = 0
        for i, img in enumerate(images, 1):
            print(f"\n{'='*40}")
            print(f"[{i}/{len(images)}]")
            print(f"{'='*40}")
            
            result = self.process_image(img)
            
            if result.get('success'):
                successful += 1
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['filename'],
                    result.get('payee', 'ERROR'),
                    result.get('issuer', 'ERROR'),
                    result.get('confidence', 0)
                ])
        
        # Summary
        print("\n" + "="*60)
        print("🎉 PROCESSING COMPLETE!")
        print("="*60)
        print(f"Total: {len(images)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {len(images)-successful}")
        print(f"📁 Results: {csv_file}")
        print("="*60)
        
        return csv_file

def main():
    print("\n" + "="*60)
    print("🤖 FINAL AI PROCESSOR - RUNNING")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python final_ai_processor.py <folder_path> [limit]")
        print("Example: python final_ai_processor.py F_23022026_010 10")
        print("\nFolders available:")
        for f in Path('.').glob('F_*'):
            print(f"  - {f}")
        sys.exit(1)
    
    folder = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    processor = FinalAIProcessor()
    processor.process_folder(folder, limit)

if __name__ == "__main__":
    main()