import csv
import argparse
from pathlib import Path
from datetime import datetime
import re
import difflib
import cv2
import numpy as np
from paddleocr import PaddleOCR
import dbf
import logging
from typing import List, Dict
import traceback

class IssuerBatchProcessorV3:
    JUNK_TERMS = {
        "NON-CASH TRANSACTION ONLY","WBO AHMEDNAGAF","THREE MONTHS", "3 MONTHS", "PLEASE SIGN", "SIGN HERE", "SIGN ABOVE", 
        "ROAD", "MUMBAI", "MAHARASHTRA", "FARIDABAD", "BEARER", "OR BEARER", "ORDER", "PAY", "NOT OVER", 
        "RS.", "RUPEES", "A/C PAYEE", "PAYEE ONLY", "ONLY", "BRANCH", "VIL", "DIST", "STATE", "PIN", 
        "CODE", "ADDRESS", "CONTACT", "PHONE", "MOBILE", "EMAIL", "GST", "PAN", "TAN", "AUTHORISED", 
        "AUTHORIZED", "SIGNATORY", "SIGNATURE", "A/C NO", "Plsse sign aboy", "vigag", "CTS CLEARING", 
        "SAVINGSAC", "PROPRIETOR", "CURRENTAC", "A/C", "PAYEE", "ONLY", "NOTOVER", "NOT OVER", "RS",
        "ACCOUNT", "PAYEE", "RUPEES", "AMOUNT", "DATE", "BEARER", "CHEQUE", "HDFC BANK LTD", "HDFC BANK", 
        "STATE BANK", "SBI", "ICICI", "AXIS", "YES", "IDFC", "KOTAK", "INDUSIND", "PNB"
    }

    BANK_TERMS = {
        "BANK", "STATE BANK", "HDFC", "ICICI", "SBI", "AXIS",
        "IDFC", "KOTAK", "INDUSIND", "PNB", "UNION BANK", "CANARA", 
        "HDFC BANK LTD", "IDFC FIRST", "RBL", "BOB", "CENTRAL BANK", 
        "CORPORATION BANK", "ALLAHABAD BANK"
    }

    COMPANY_HINTS = {"LTD", "LIMITED", "PVT", "PRIVATE", "LLP", "CO", "CORP"}
    
    # Common Indian name components
    COMMON_NAME_PARTS = {
        "KUMAR", "SINGH", "SHARMA", "VERMA", "GUPTA", "PATEL", "SHAH", "MEHTA",
        "JOSHI", "PANDEY", "TIWARI", "MISHRA", "DUBEY", "TRIPATHI", "CHOUDHARY",
        "CHAUDHARY", "YADAV", "JAISWAL", "DAS", "BANERJEE", "CHATTERJEE",
        "MUKHERJEE", "SARKAR", "BOSE", "GHOSH", "RAO", "REDDY", "KUMARI",
        "DEVI", "PRASAD", "RAM", "LAL", "AHMED", "KHAN", "ANSARI", "SIDDIQUI",
        "ALI", "HUSSAIN", "RAJ", "SONI", "JAIN", "AGARWAL", "GOYAL", "MITTAL",
        "MALIK", "KAUR", "GILL", "DHILLON", "BRAR", "SANDHU", "STORE", "STORES",
        "MART", "TRADERS", "ENTERPRISES", "AGENCIES", "BROTHERS", "AND", "CO",
        "COMPANY", "INDUSTRIES", "PHARMA", "MEDICAL", "HOSPITAL", "CLINIC"
    }
    
    FIXED_OPR_NO = "AS601"
    FIXED_FILE_MARK = False
    
    stats = {
        'processed': 0,
        'valid_names': 0,
        'xxx': 0,
        'for_pattern': 0,
        'garbage_rejected': 0
    }
    
    def __init__(self, args):
        self.input_dir = Path(args.input_dir)
        self.limit = args.limit
        self.dbf_path = Path(args.dbf_path)
        self.threshold = args.threshold
        self.output_dbf_path = None

        print("Initializing PaddleOCR...")
        # Use better OCR settings
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
            det_db_thresh=0.3,  # Lower threshold for better detection
            det_db_box_thresh=0.2,  # More boxes
            rec_db_thresh=0.3  # Lower recognition threshold
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("issuer_output")
        output_dir.mkdir(exist_ok=True)
        
        self.output_dbf_path = output_dir / f"ISSUER_RESULTS_{timestamp}.dbf"
        
        self.output_table = dbf.Table(
            str(self.output_dbf_path),
            'IMAGE_FILE C(50); DRAWER_NM C(100); OPR_NO C(20); FILE_MARK L; CONFIDENCE N(6,4); STATUS C(20)',
        )
        self.output_table.open(mode=dbf.READ_WRITE)

        self.csv_path = output_dir / f"ISSUER_RESULTS_{timestamp}.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "issuer_name", "opr_no", "file_mark", "confidence", "status", "method"])

        if not self.dbf_path.exists():
            raise FileNotFoundError("DBF file not found.")

        self.table = dbf.Table(str(self.dbf_path))
        self.table.open(mode=dbf.READ_WRITE)

        self.drawer_dict = self.build_drawer_dict()
        print(f"Drawer dictionary loaded: {len(self.drawer_dict)} names")

    def clean_text(self, text):
        """More careful text cleaning"""
        if not text:
            return "XXX"

        # Convert to uppercase
        text = text.upper()
        
        # Fix common OCR errors
        text = text.replace("0", "O").replace("1", "I").replace("|", "I")
        text = text.replace("5", "S").replace("6", "G").replace("8", "B")
        
        # Keep important characters for business names
        text = re.sub(r"[^A-Z0-9 .,&/-]", " ", text)
        
        # Remove extra spaces
        text = " ".join(text.split())
        
        return text.strip()

    def is_junk_text(self, text):
        """Check if text is junk - more lenient"""
        if not text or len(text) < 2:
            return True

        text_upper = text.upper()

        # Direct junk match - only reject exact matches, not partial
        for term in self.JUNK_TERMS:
            if text_upper == term or text_upper.startswith(term + " "):
                return True

        return False

    def is_valid_drawer(self, text):
        """More lenient validation for drawer names"""
        if self.is_junk_text(text):
            return False
        
        # Don't automatically reject bank terms - some companies have "BANK" in name
        # Just check if it's ONLY a bank term
        if text in self.BANK_TERMS:
            return False

        # Basic checks
        if len(text) < 3:  # Very short
            return False

        # Count letters and digits
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        total = len(text)
        
        if total == 0:
            return False

        # If it's mostly digits, reject
        if digits > letters and letters < 3:
            return False

        # If it has at least 3 letters, consider it valid
        if letters >= 3:
            return True

        return False

    def build_drawer_dict(self):
        """Build dictionary of known drawer names"""
        names = set()
        for record in list(self.table):
            drawer = self.clean_text(str(record.DRAWER_NM).strip())
            if drawer and drawer != "XXX" and len(drawer) > 2:
                names.add(drawer)
        return sorted(names)

    def match_name(self, text):
        """Match name against dictionary"""
        if not text or len(text) < 3:
            return None, 0.0

        text = text.upper()

        # Exact match
        if text in self.drawer_dict:
            return text, 1.0

        # Try partial matching
        best_match = None
        best_score = 0.0
        
        for drawer in self.drawer_dict:
            # Check if text is contained in drawer
            if text in drawer:
                similarity = len(text) / len(drawer)
                if similarity > 0.6 and similarity > best_score:
                    best_match = drawer
                    best_score = similarity
            
            # Check if drawer is contained in text
            elif drawer in text:
                similarity = len(drawer) / len(text)
                if similarity > 0.6 and similarity > best_score:
                    best_match = drawer
                    best_score = similarity
            
            # Fuzzy match
            else:
                similarity = difflib.SequenceMatcher(None, text, drawer).ratio()
                if similarity > 0.7 and similarity > best_score:
                    best_match = drawer
                    best_score = similarity

        return best_match, best_score

    def extract_issuer(self, ocr_result):
        """Extract issuer name from OCR results"""
        if not ocr_result or not ocr_result[0]:
            return "XXX", 0.0
        
        candidates = []

        for line in ocr_result[0]:
            if len(line) < 2:
                continue

            raw_text = line[1][0].strip()
            conf = float(line[1][1])

            # Skip very low confidence
            if conf < 0.3:
                continue

            text = self.clean_text(raw_text)
            
            # Special handling for FOR patterns
            text_upper = text.upper()
            for marker in ["FOR ", "F/O ", "F/", "PROP ", "PROP."]:
                if marker in text_upper:
                    parts = text_upper.split(marker, 1)
                    if len(parts) > 1:
                        potential_name = parts[1].strip()
                        if potential_name and len(potential_name) > 2:
                            # Boost confidence for FOR pattern
                            candidates.append((conf + 0.2, potential_name, "FOR_PATTERN"))
                            self.stats['for_pattern'] += 1
                            continue
            
            # Regular validation
            if not self.is_valid_drawer(text):
                continue

            matched_name, similarity = self.match_name(text)
            final_text = matched_name if matched_name else text

            # Calculate score
            score = conf
            
            # Boost for dictionary match
            if matched_name:
                score += 0.2 * similarity

            # Boost for reasonable length
            if 5 <= len(final_text) <= 30:
                score += 0.1

            candidates.append((score, final_text, "OCR_NORMAL"))

        if not candidates:
            return "XXX", 0.0

        # Sort by score and return best
        candidates.sort(reverse=True, key=lambda x: x[0])
        best_score, best_text, method = candidates[0]

        return best_text, best_score

    def process_image(self, image_path):
        """Process single image with multiple ROI attempts"""
        img = cv2.imread(str(image_path))
        if img is None:
            return "XXX", 0.0

        h, w, _ = img.shape

        # Try multiple ROIs to find issuer name
        rois = [
            # Standard issuer area
            (int(h * 0.30), int(h * 0.85), int(w * 0.30), w),  # Wider area
            (int(h * 0.40), int(h * 0.90), int(w * 0.35), w),  # Original
            (int(h * 0.35), int(h * 0.80), int(w * 0.25), w),  # Higher up
            (int(h * 0.45), int(h * 0.95), int(w * 0.40), w),  # Lower
        ]

        best_result = "XXX"
        best_confidence = 0.0
        best_method = "NONE"

        for i, (y1, y2, x1, x2) in enumerate(rois):
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Preprocess ROI
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Try different preprocessing techniques
            preprocessed_images = [
                gray,  # Original
                cv2.equalizeHist(gray),  # Histogram equalization
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2),  # Adaptive threshold
            ]

            for proc_img in preprocessed_images:
                result = self.ocr.ocr(proc_img, cls=True)
                if result and result[0]:
                    text, conf = self.extract_issuer(result)
                    
                    # Check if this is better than previous results
                    if conf > best_confidence and text != "XXX":
                        # Additional validation for business names
                        if len(text) >= 3:
                            best_result = text
                            best_confidence = conf
                            best_method = f"ROI_{i+1}"

        return best_result, best_confidence, best_method

    def looks_like_real_name(self, text):
        """Simple validation - much more lenient"""
        if not text or text == "XXX":
            return False
        
        # At least 3 characters
        if len(text) < 3:
            return False
        
        # At least 2 letters
        letters = sum(c.isalpha() for c in text)
        if letters < 2:
            return False
        
        # Not too many digits
        digits = sum(c.isdigit() for c in text)
        if digits > letters and letters < 3:
            return False
        
        return True

    def update_original_dbf(self, image_name, issuer):
        """Update original DBF"""
        image_base = Path(image_name).stem.strip().lower()
        
        for record in list(self.table):
            dbf_image = str(record["IMAGE_FILE"]).strip().lower()
            dbf_base = Path(dbf_image).stem.strip().lower()
            
            if dbf_base == image_base:
                with record:
                    record.DRAWER_NM = issuer[:50]
                    try:
                        record.OPR_NO = self.FIXED_OPR_NO
                    except Exception:
                        pass
                    try:
                        record.FILE_MARK = self.FIXED_FILE_MARK
                    except Exception:
                        pass
                return True
        return False

    def append_to_output_dbf(self, image_name, issuer, confidence, status):
        """Append to output DBF"""
        if len(status) > 20:
            status = status[:20]
           
        self.output_table.append({
            'IMAGE_FILE': str(image_name),
            'DRAWER_NM': str(issuer)[:50],
            'OPR_NO': self.FIXED_OPR_NO,
            'FILE_MARK': self.FIXED_FILE_MARK,
            'CONFIDENCE': float(confidence),
            'STATUS': str(status)
        })
        return True

    def run(self):
        if not self.input_dir.exists():
            print("Input directory not found.")
            return

        images = sorted(
            p for p in self.input_dir.iterdir()
            if p.suffix.lower() in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
        )

        if self.limit:
            images = images[:self.limit]

        print(f"Processing {len(images)} images...")
        print(f"Confidence threshold: {self.threshold}")
        
        batch_size = 50
        total = len(images)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_images = images[start:end]

            print(f"\nProcessing batch {start + 1} to {end}...\n")

            # Reinitialize OCR for each batch
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False
            )

            for img_path in batch_images:
                try:
                    issuer, confidence, method = self.process_image(img_path)
                    
                    self.stats['processed'] += 1
                    
                    # Determine status based on confidence and validation
                    if confidence >= self.threshold and self.looks_like_real_name(issuer):
                        status = "AUTO-UPDATED"
                        self.stats['valid_names'] += 1
                    elif issuer != "XXX" and confidence > 0:
                        status = "REVIEW"
                        self.stats['valid_names'] += 1
                    else:
                        issuer = "XXX"
                        status = "REJECTED"
                        self.stats['xxx'] += 1

                    # Append to output DBF
                    self.append_to_output_dbf(img_path.name, issuer, confidence, status)

                    # Update original DBF
                    self.update_original_dbf(img_path.name, issuer)

                    # Write to CSV
                    with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            img_path.name, issuer, self.FIXED_OPR_NO, 
                            self.FIXED_FILE_MARK, round(confidence, 4), status, method
                        ])

                    print(f"SUMMARY: {img_path.name} -> Issuer: {issuer} ({method}) [{round(confidence, 2)}] {status}")

                except Exception as e:
                    print(f"Error processing {img_path.name}: {e}")
                    traceback.print_exc()

        # Close tables
        self.output_table.close()
        self.table.close()
        
        # Print statistics
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE!")
        print(f"{'='*50}")
        print(f"Output DBF file: {self.output_dbf_path}")
        print(f"\nFINAL STATISTICS:")
        print(f"  Total processed: {self.stats['processed']}")
        print(f"  Valid names extracted: {self.stats['valid_names']}")
        print(f"  FOR pattern matches: {self.stats['for_pattern']}")
        print(f"  XXX returned: {self.stats['xxx']}")
        if self.stats['processed'] > 0:
            accuracy = (self.stats['valid_names'] / self.stats['processed']) * 100
            print(f"  Success rate: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--dbf-path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.70)  # Lower default
    parser.add_argument("--use-gpu", action="store_true")
    
    args = parser.parse_args()
    processor = IssuerBatchProcessorV3(args)
    processor.run()