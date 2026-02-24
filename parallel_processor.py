#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel Processing with GPU Support - WITH ISSUER EXTRACTION (FIXED DBF VERSION)
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import csv
import re
import dbf

from ocr_engine import OCREngine
from tiff_processor import get_tiff_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= PAYEE WRITER =================

class PayeeWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.output_dir / f"PAYEE_NAMES_{timestamp}.csv"
        self.processed_files = set()

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Payee Name', 'Filename'])

        print(f"\n🔴 PAYEE FILE CREATED: {self.csv_path}\n")

    def write(self, payee, filename):
        if filename in self.processed_files:
            return False

        self.processed_files.add(filename)

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([payee, filename])

        return True


# ================= ISSUER WRITER =================

class IssuerWriter:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.output_dir / f"ISSUER_NAMES_{timestamp}.csv"

        self.processed_files = set()

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Issuer Name', 'Filename'])

        print(f"\n🔵 ISSUER FILE CREATED: {self.csv_path}\n")

    def extract_issuer_from_text(self, text):
        if not text:
            return None

        text = text.upper()

        patterns = [
            r'([A-Z][A-Z\s\.]{3,50}?)\s+\d{10,20}',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+(?:CURRENT|SAVINGS)\s+A\/C',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+(?:HDFC|ICICI|SBI|YES|AXIS)\s+BANK',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)

                if len(name) > 3 and not any(x in name for x in ['RUPEES', 'BANK']):
                    return name

        return None

    def write_result(self, ocr_result):
        filename = Path(ocr_result.get('image_path', '')).name

        if filename in self.processed_files:
            return False

        self.processed_files.add(filename)

        text = ocr_result.get('full_text', '')
        issuer = self.extract_issuer_from_text(text) or "UNKNOWN"

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([issuer, filename])

        return True


# ================= DBF WRITER =================

class DBFWriterWithAI:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dbf_path = self.output_dir / f"CLEAN_ISSUER_DATA_{timestamp}.dbf"

        self.table = dbf.Table(
            str(self.dbf_path),
            'ISSUER_NAME C(100); FILENAME C(50); CONFIDENCE N(5,2); PROCESS_DATE C(19)'
        )

        self.table.open(mode=dbf.READ_WRITE)

        print(f"\n💾 DBF FILE CREATED: {self.dbf_path}\n")

        self.processed_files = set()

    def write_result(self, ocr_result, issuer_name):
        filename = Path(ocr_result.get('image_path', '')).name

        if filename in self.processed_files:
            return False

        self.processed_files.add(filename)

        confidence = float(ocr_result.get('confidence', 0.0))
        process_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.table.append((
            issuer_name.title(),
            filename,
            round(confidence, 2),
            process_date
        ))

        return True


# ================= PARALLEL PROCESSOR =================
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

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.gpu_mem = gpu_mem

        self.payee_writer = PayeeWriter(self.output_dir)
        self.issuer_writer = IssuerWriter(self.output_dir)
        self.dbf_writer = DBFWriterWithAI(self.output_dir)

        if self.use_gpu:
            logger.info(f"✅ GPU Enabled (ID: {self.gpu_id})")
    @staticmethod
    def _process_chunk(image_paths, use_gpu, gpu_id):

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

        results = self._process_chunk(image_paths, self.use_gpu, self.gpu_id)

        for result in results:
            if result.get('success'):

                filename = Path(result.get('image_path')).name

                payee = result.get('payee_cleaned', 'UNKNOWN')
                self.payee_writer.write(payee, filename)

                self.issuer_writer.write_result(result)

                issuer = self.issuer_writer.extract_issuer_from_text(
                    result.get('full_text', '')
                ) or "UNKNOWN"

                self.dbf_writer.write_result(result, issuer)

        logger.info("✅ Processing Complete")