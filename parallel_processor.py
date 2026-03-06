#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel Processing with GPU Support - Professional Version
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import csv
import re
import dbf

from ocr_engine import OCREngine
from tiff_processor import get_tiff_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IssuerExtractor:
    """Professional issuer name extractor"""
    
    def __init__(self):
        self.bank_names = [
            'STATE BANK OF INDIA', 'HDFC BANK', 'ICICI BANK', 'AXIS BANK',
            'PUNJAB NATIONAL BANK', 'BANK OF BARODA', 'CANARA BANK',
            'UNION BANK OF INDIA', 'KOTAK MAHINDRA BANK', 'INDUSIND BANK',
            'YES BANK', 'IDBI BANK', 'BANK OF INDIA', 'CENTRAL BANK OF INDIA'
        ]
        
        self.business_identifiers = [
            'PVT LTD', 'PRIVATE LIMITED', 'LIMITED', 'LTD', 'ENTERPRISES',
            'INDUSTRIES', 'TRADERS', 'COMPANY', 'CORPORATION'
        ]
        
        self.ignore_words = {
            'RUPEES', 'AMOUNT', 'TOTAL', 'PAY', 'BEARER', 'A/C', 'ACCOUNT',
            'BRANCH', 'IFSC', 'CODE', 'DATE', 'VALID', 'MONTHS', 'FROM',
            'AUTHORISED', 'SIGNATORY', 'SIGNATURE', 'SIGN'
        }
    
    def extract_from_text(self, text: str) -> str:
        """Extract issuer name from OCR text"""
        if not text or len(text) < 20:
            return "UNKNOWN"
        
        text_upper = text.upper()
        
        # Direct bank matching
        for bank in self.bank_names:
            if bank in text_upper:
                return bank
        
        # FOR pattern
        if 'FOR' in text_upper:
            parts = text_upper.split('FOR')
            if len(parts) > 1:
                candidate = parts[-1].strip()
                for delimiter in ['AUTHORISED', 'SIGN', 'ACCOUNT', 'A/C', '\n']:
                    if delimiter in candidate:
                        candidate = candidate.split(delimiter)[0]
                
                words = candidate.split()
                if 1 <= len(words) <= 5:
                    cleaned = self._clean_name(' '.join(words))
                    if cleaned != "UNKNOWN":
                        return cleaned
        
        # Business names
        lines = text_upper.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 15:
                for identifier in self.business_identifiers:
                    if identifier in line:
                        cleaned = self._clean_name(line)
                        if cleaned != "UNKNOWN":
                            return cleaned
        
        # Last lines
        valid_lines = [l.strip() for l in lines if len(l.strip()) > 10]
        for line in valid_lines[-3:]:
            if line.isupper() and ' ' in line:
                if not any(word in line for word in self.ignore_words):
                    cleaned = self._clean_name(line)
                    if cleaned != "UNKNOWN":
                        return cleaned
        
        return "UNKNOWN"
    
    def _clean_name(self, name: str) -> str:
        """Clean extracted name"""
        if not name:
            return "UNKNOWN"
        
        name = re.sub(r'[^\w\s]', ' ', name)
        name = ' '.join(name.split())
        
        if len(name) < 5:
            return "UNKNOWN"
        
        return name.title()


class PayeeWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed = set()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.output_dir / f"PAYEE_NAMES_{timestamp}.csv"
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Payee Name', 'Filename'])
        
        print(f"\n🔴 PAYEE FILE CREATED: {self.csv_path}")

    def write(self, payee, filename):
        if filename in self.processed:
            return False
        
        self.processed.add(filename)
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([payee, filename])
        
        return True


class IssuerWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed = set()
        self.extractor = IssuerExtractor()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.output_dir / f"ISSUER_NAMES_{timestamp}.csv"
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Issuer Name', 'Filename', 'Confidence'])
        
        print(f"\n🔵 ISSUER FILE CREATED: {self.csv_path}")
    
    def extract_issuer_from_text(self, text):
        return self.extractor.extract_from_text(text)
    
    def write_result(self, ocr_result):
        filename = Path(ocr_result.get('image_path', '')).name
        
        if filename in self.processed:
            return False
        
        self.processed.add(filename)
        
        text = ocr_result.get('full_text', '')
        confidence = ocr_result.get('confidence', 0)
        
        issuer = self.extractor.extract_from_text(text)
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([issuer, filename, confidence])
        
        return True


class DBFWriterWithAI:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed = set()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dbf_path = self.output_dir / f"CLEAN_ISSUER_DATA_{timestamp}.dbf"
        
        # DBF with 10-char field names
        self.table = dbf.Table(
            str(self.dbf_path),
            'ISSUER C(100); FNAME C(50); CONFID N(5,2); PROCDT C(19)'
        )
        self.table.open(mode=dbf.READ_WRITE)
        
        print(f"\n💾 DBF FILE CREATED: {self.dbf_path}")
        print(f"   Fields: ISSUER, FNAME, CONFID, PROCDT")

    def write_result(self, ocr_result, issuer_name):
        filename = Path(ocr_result.get('image_path', '')).name
        
        if filename in self.processed:
            return False
        
        self.processed.add(filename)
        
        confidence = float(ocr_result.get('confidence', 0.0))
        process_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.table.append((
            issuer_name[:100],  # Truncate if too long
            filename,
            round(confidence, 2),
            process_date
        ))
        
        return True


class ParallelProcessor:
    def __init__(
        self,
        num_cpus: int = 1,
        processes_per_cpu: int = 2,
        cpu_threads_per_process: int = 1,
        output_dir: str = "./results",
        use_gpu: bool = False,
        gpu_id: int = 0,
        gpu_mem: int = 4000
    ):
        self.num_cpus = num_cpus
        self.processes_per_cpu = processes_per_cpu
        self.cpu_threads_per_process = cpu_threads_per_process
        
        # Create batch directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir) / f"batch_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.gpu_mem = gpu_mem
        
        # Initialize writers
        self.payee_writer = PayeeWriter(self.output_dir)
        self.issuer_writer = IssuerWriter(self.output_dir)
        self.dbf_writer = DBFWriterWithAI(self.output_dir)
        
        if self.use_gpu:
            logger.info(f"GPU Enabled (ID: {self.gpu_id})")
    
    @staticmethod
    def _process_chunk(image_paths, use_gpu, gpu_id):
        """Process a chunk of images"""
        ocr_engine = OCREngine(use_gpu=use_gpu, gpu_id=gpu_id)
        tiff_processor = get_tiff_processor(scale_percent=50)
        
        results = []
        for image_path in image_paths:
            try:
                img, metadata = tiff_processor.preprocess(image_path)
                result = ocr_engine.process_cheque(img, str(image_path))
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def run(self, image_paths):
        """Process a batch of images"""
        successful = 0
        failed = 0
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        # Process images
        results = self._process_chunk(image_paths, self.use_gpu, self.gpu_id)
        
        for result in results:
            if result.get('success'):
                successful += 1
                filename = Path(result.get('image_path')).name
                
                # Write payee
                payee = result.get('payee_cleaned', 'UNKNOWN')
                self.payee_writer.write(payee, filename)
                
                # Write issuer
                self.issuer_writer.write_result(result)
                
                # Write DBF
                issuer = self.issuer_writer.extract_issuer_from_text(
                    result.get('full_text', '')
                ) or "UNKNOWN"
                self.dbf_writer.write_result(result, issuer)
                
                logger.info(f"✓ {filename}: Issuer={issuer}")
            else:
                failed += 1
                logger.error(f"✗ Failed: {result.get('image_path')}")
        
        logger.info(f"Batch complete: {successful} successful, {failed} failed")
        
        return {
            'successful': successful,
            'failed': failed,
            'total': len(image_paths)
        }