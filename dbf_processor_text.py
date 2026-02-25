#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Powered DBF Issuer Name Processor - TEXT FILE VERSION
Supports: Cheque images (.tiff, .tif, .jpg, .png) + Existing DBF
Outputs: CSV + TEXT file with AI-corrected issuer names
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# DBF handling
from dbfread import DBF

# OCR & TIFF processing (your modules)
from ocr_engine import OCREngine
from tiff_processor import get_tiff_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AIIssuerExtractor:
    """AI-powered issuer name extractor with spelling correction"""

    def __init__(self, use_gpu=True, gpu_id=0):
        self.ocr_engine = OCREngine(use_gpu=use_gpu, gpu_id=gpu_id)
        self.tiff_processor = get_tiff_processor()
        # Predefined spelling corrections
        self.spelling_corrections = {
            'STEELNDUSTRIES': 'STEEL INDUSTRIES',
            'STEELHDUSTRIES': 'STEEL INDUSTRIES',
            'STEELIHDUSTRIES': 'STEEL INDUSTRIES',
            'STEELIINDUSTRIES': 'STEEL INDUSTRIES',
            'STEELINDUSTRIES': 'STEEL INDUSTRIES',
            'STEL INDUSTWUES': 'STEEL INDUSTRIES',
            'DEVAPRLIANGES': 'DEV APPLIANCES',
            'APRLIANGES': 'APPLIANCES',
            'COAL': 'GOAL',
            'SURYA COAL': 'SURYA GOAL',
            'ENTENPES': 'ENTERPRISES',
            'ENTSEPS': 'ENTERPRISES',
            'KISPA': 'KIXPA',
            'ARIHANTINEOCOO': 'ARIHANT NEOCO',
            'CURUGRAM': 'GURUGRAM',
            'FASTNERS': 'FASTNERS',
            'NARENDRA FASTNERS': 'NARENDRA FASTNERS',
            'HARJEET KAUR': 'HARJEET KAUR',
            'SATBANT KAUR': 'SATBANT KAUR',
            'NEELAM JINDAL': 'NEELAM JINDAL',
            'MOHD QURBAN': 'MOHD QURBAN',
            'DUDHI INDUSTRIES': 'DUDHI INDUSTRIES',
            'ROHIN': 'ROHIT',
            'POLYPLAST': 'POLYPLAST PVT LTD',
            'BATRA SCREW': 'BATRA SCREW INDUSTRIES',
        }
        self.garbage_words = [
            'RUPEES', 'WOTFTHO', 'BEARER', 'QRBEARER', 'RQR', 
            'XXX', 'RUPEE', 'RS', 'AMOUNT', 'TOTAL', 'PAY',
            'OR BEARER', 'A/C', 'ACCOUNT', 'BANK', 'BRANCH',
            'IFSC', 'CODE', 'DATE', 'VALID', 'MONTHS', 'FROM',
            'THE', 'OF', 'ISSUE', 'ONLY', 'AND', 'FOR', 'TO'
        ]

    def extract_issuer_from_text(self, text):
        """Extract issuer name using AI patterns"""
        if not text:
            return None
        text = text.upper()
        # Pattern 1: AFTER 'FOR'
        if 'FOR' in text:
            parts = text.split('FOR')
            if len(parts) > 1:
                candidate = parts[1].strip()
                for end_marker in [' AUTH', ' SIGN', ' I/', ' D ', ' AUTHORISED', ' AUTHORIZED', ' A/C', ' ACCOUNT']:
                    if end_marker in candidate:
                        candidate = candidate.split(end_marker)[0]
                candidate = candidate.split('\n')[0]
                if len(candidate) > 3:
                    return self.ai_correct_spelling(candidate)
        # Pattern 2: Before auth signatory
        auth_patterns = [
            r'([A-Z][A-Z\s\.]{3,50}?)\s+Auth\s*Signatory',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+Authorised\s+Signatory',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+AUTHORISED\s+SIGNATORY',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+Signature',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+Sign',
        ]
        for pattern in auth_patterns:
            match = re.search(pattern, text)
            if match:
                return self.ai_correct_spelling(match.group(1))
        # Pattern 3: Before account numbers
        account_patterns = [
            r'([A-Z][A-Z\s\.]{3,50}?)\s+\d+\s+A\/C\.?',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+A\/C\.?\s+NO\.?',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+\d{10,20}',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+CURRENT\s+A\/C',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+SAVINGS\s+A\/C',
        ]
        for pattern in account_patterns:
            match = re.search(pattern, text)
            if match:
                return self.ai_correct_spelling(match.group(1))
        # Pattern 4: Last lines
        lines = text.split('\n')
        for line in reversed(lines[-10:]):
            line = line.strip()
            if len(line) > 5 and line.isupper() and not any(g in line for g in self.garbage_words):
                return self.ai_correct_spelling(line)
        return None

    def ai_correct_spelling(self, name):
        if not name:
            return "UNKNOWN"
        name = name.upper()
        for wrong, correct in self.spelling_corrections.items():
            name = name.replace(wrong, correct)
        for word in self.garbage_words:
            name = name.replace(word, '')
        name = ' '.join(name.split())
        import re
        name = re.sub(r'[^A-Z\s]', '', name)
        return name.title() if len(name) >= 3 else "UNKNOWN"

    def process_cheque(self, image_path):
        try:
            img, metadata = self.tiff_processor.preprocess(image_path)
            result = self.ocr_engine.process_cheque(img, str(image_path))
            if not result.get('success'):
                return {'filename': Path(image_path).name, 'issuer_raw': None, 'issuer_clean': 'UNKNOWN', 'confidence': 0, 'error': result.get('error', 'OCR failed')}
            text = result.get('full_text', '')
            issuer_raw = self.extract_issuer_from_text(text)
            return {'filename': Path(image_path).name, 'issuer_raw': issuer_raw, 'issuer_clean': issuer_raw if issuer_raw else 'UNKNOWN', 'confidence': result.get('confidence', 0), 'error': None}
        except Exception as e:
            return {'filename': Path(image_path).name, 'issuer_raw': None, 'issuer_clean': 'UNKNOWN', 'confidence': 0, 'error': str(e)}


class DBFIssuerProcessor:
    """Process DBF file and add AI-corrected issuer names, auto-search images in DBF folder if needed"""

    def __init__(self, template_dbf_path, output_text_path=None):
        self.template_path = Path(template_dbf_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template DBF not found: {template_dbf_path}")

        # Create output text file path (same name but .txt extension)
        if output_text_path:
            self.output_path = Path(output_text_path)
        else:
            self.output_path = self.template_path.parent / f"updated_{self.template_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        self.template = DBF(str(self.template_path))
        self.field_names = self.template.field_names
        self.field_specs = [(f.name, f.type, f.length, getattr(f, 'decimal_count', 0)) for f in self.template.fields]
        self.extractor = AIIssuerExtractor(use_gpu=True)

        # Default folder for images is same as DBF folder
        self.default_image_dir = self.template_path.parent

        logger.info(f"📁 Template DBF: {self.template_path}")
        logger.info(f"📁 Output TEXT file: {self.output_path}")
        logger.info(f"📊 Fields: {', '.join(self.field_names)}")

    def find_matching_images(self, image_dir=None):
        """Find matching images for DBF records"""
        folder = Path(image_dir) if image_dir else self.default_image_dir
        image_files = {}
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.jpg', '*.jpeg', '*.png']:
            for img_path in folder.glob(ext):
                image_files[img_path.name] = img_path
                image_files[img_path.stem] = img_path
                if img_path.stem.startswith('P_'):
                    image_files[img_path.stem[2:]] = img_path
        return image_files

    def process(self, image_dir, filename_field='IMAGE_FILE', issuer_field='DRAWER_NAME'):
        image_files = self.find_matching_images(image_dir)
        logger.info(f"📸 Found {len(image_files)} image files indexed")
        records = list(self.template)
        logger.info(f"📊 Found {len(records)} DBF records")

        updated_records, matched_count, extracted_count = [], 0, 0
        
        # Process all records
        for i, record in enumerate(records):
            record_dict = dict(record)
            filename = str(record_dict.get(filename_field, '')).strip()
            matched_image = None
            filename_lower = filename.lower()
            
            for key, img_path in image_files.items():
                if key.lower() == filename_lower or Path(filename).stem.lower() == key.lower():
                    matched_image = img_path
                    break
                    
            issuer_name = "XXX"
            if matched_image:
                matched_count += 1
                logger.info(f"🔍 Processing [{i+1}/{len(records)}]: {matched_image.name}")
                result = self.extractor.process_cheque(matched_image)
                issuer_name = result['issuer_clean']
                if result['error']:
                    logger.warning(f"⚠️  Error: {result['error']}")
                else:
                    extracted_count += 1
                    logger.info(f"   ✅ Issuer: {issuer_name}")
            else:
                logger.warning(f"⚠️  No image found for: {filename}")
            
            # Add issuer name to record
            record_dict[issuer_field] = issuer_name
            updated_records.append(record_dict)

        # Create DataFrame and save CSV
        df = pd.DataFrame(updated_records)
        csv_path = self.output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"✅ CSV file created: {csv_path}")

        # Create TEXT file (Notepad-friendly)
        self.create_text_file(updated_records, filename_field, issuer_field)
        
        # Also create a simple summary text file
        self.create_summary_file(updated_records, filename_field, issuer_field, matched_count, extracted_count)

        logger.info("="*60)
        logger.info("🎉 PROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"📊 Total DBF records: {len(records)}")
        logger.info(f"📸 Images matched: {matched_count}")
        logger.info(f"✅ Issuers extracted: {extracted_count}")
        logger.info(f"❌ Failed: {matched_count - extracted_count}")
        logger.info(f"📄 Output files:")
        logger.info(f"   - CSV: {csv_path}")
        logger.info(f"   - TEXT: {self.output_path}")
        logger.info("="*60)
        
        return {
            'total': len(records), 
            'matched': matched_count, 
            'extracted': extracted_count, 
            'csv_output': str(csv_path),
            'text_output': str(self.output_path)
        }
    
    def create_text_file(self, records, filename_field, issuer_field):
        """Create a simple text file with the results (Notepad compatible)"""
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("="*80 + "\n")
                f.write("AI-POWERED DBF ISSUER NAME PROCESSOR - RESULTS\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                # Write summary
                f.write("SUMMARY\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Records Processed: {len(records)}\n")
                f.write(f"Output Field: {issuer_field}\n\n")
                
                # Write detailed results
                f.write("DETAILED RESULTS\n")
                f.write("-"*80 + "\n")
                
                # Create a formatted table
                header = f"{'Sr No.':<8} {'Filename':<30} {'Issuer Name':<40}\n"
                f.write(header)
                f.write("-"*80 + "\n")
                
                for idx, record in enumerate(records, 1):
                    filename = str(record.get(filename_field, 'N/A'))[:30]
                    issuer = str(record.get(issuer_field, 'XXX'))[:40]
                    f.write(f"{idx:<8} {filename:<30} {issuer:<40}\n")
                
                # Write footer
                f.write("\n" + "="*80 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*80 + "\n")
            
            logger.info(f"✅ Text file created: {self.output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create text file: {e}")
    
    def create_summary_file(self, records, filename_field, issuer_field, matched_count, extracted_count):
        """Create a simple summary text file with just the essential data"""
        try:
            summary_path = self.output_path.parent / f"summary_{self.output_path.name}"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("FILENAME\tISSUER_NAME\n")
                f.write("-"*50 + "\n")
                for record in records:
                    filename = str(record.get(filename_field, 'N/A'))
                    issuer = str(record.get(issuer_field, 'XXX'))
                    f.write(f"{filename}\t{issuer}\n")
                
                f.write("\n" + "="*50 + "\n")
                f.write(f"Total: {len(records)} records\n")
                f.write(f"Matched: {matched_count}\n")
                f.write(f"Extracted: {extracted_count}\n")
            
            logger.info(f"✅ Summary text file created: {summary_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create summary file: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI DBF Issuer Name Processor - Creates TEXT files instead of DBF')
    parser.add_argument('--dbf', '-d', required=True, help='Template DBF file')
    parser.add_argument('--images', '-i', required=True, help='Directory with cheque images')
    parser.add_argument('--output', '-o', help='Output TEXT file (optional)')
    parser.add_argument('--filename-field', default='IMAGE_FILE', help='Field name for filename (default: IMAGE_FILE)')
    parser.add_argument('--issuer-field', default='DRAWER_NAME', help='Field name for drawer name (default: DRAWER_NAME)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    args = parser.parse_args()

    processor = DBFIssuerProcessor(args.dbf, args.output)
    if args.no_gpu:
        processor.extractor.ocr_engine.use_gpu = False
    processor.process(
        image_dir=args.images, 
        filename_field=args.filename_field, 
        issuer_field=args.issuer_field
    )


if __name__ == "__main__":
    main()