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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict, Optional
import gc
import traceback

class IssuerBatchProcessorV3:
    JUNK_TERMS = {
        "NON-CASH TRANSACTION ONLY","WBO AHMEDNAGAF","THREE MONTHS", "3 MONTHS", "PLEASE SIGN", "SIGN HERE", "SIGN ABOVE", "ROAD", "MUMBAI", "MAHARASHTRA", "FARIDABAD",
        "BEARER", "OR BEARER", "ORDER", "PAY", "NOT OVER", "NOT OVER", "RS.", "RUPEES", "A/C PAYEE", "PAYEE ONLY",
        "ONLY", "BRANCH", "VIL", "DIST", "STATE", "PIN", "CODE", "ADDRESS", "CONTACT", "PHONE", "MOBILE", "EMAIL", "GST", "PAN", "TAN",
        "AUTHORISED", "AUTHORIZED", "SIGNATORY", "SIGNATURE", "A/C NO", "Plsse sign aboy", "vigag", "CTS CLEARING", "SAVINGSAC", "PROPRIETOR", "CURRENTAC", "A/C", "PAYEE", "ONLY", "NOTOVER", "NOT OVER", "RS",
        "ACCOUNT", "PAYEE", "RUPEES", "AMOUNT", "DATE", "BEARER", "CHEQUE", "HDFC BANK LTD", "HDFC BANK", "STATE BANK", "SBI", "ICICI", "AXIS", "YES", "IDFC", "KOTAK", "INDUSIND", "PNB"
    }

    BANK_TERMS = {
        "BANK", "STATE BANK", "HDFC", "ICICI", "SBI", "AXIS",
        "IDFC", "KOTAK", "INDUSIND", "PNB", "UNION BANK", "CANARA", "HDFC BANK LTD", "IDFC FIRST", "RBL", "BOB", "CENTRAL BANK", "CORPORATION BANK", "ALLAHABAD BANK"
    }

    COMPANY_HINTS = {"LTD", "LIMITED", "PVT", "PRIVATE", "LLP", "CO", "CORP"}
    
    # ADDED: Common Indian name components for validation
    COMMON_NAME_PARTS = {
        "KUMAR", "SINGH", "SHARMA", "VERMA", "GUPTA", "PATEL", "SHAH", "MEHTA",
        "JOSHI", "PANDEY", "TIWARI", "MISHRA", "DUBEY", "TRIPATHI", "CHOUDHARY",
        "CHAUDHARY", "YADAV", "JAISWAL", "DAS", "BANERJEE", "CHATTERJEE",
        "MUKHERJEE", "SARKAR", "BOSE", "GHOSH", "RAO", "REDDY", "KUMARI",
        "DEVI", "PRASAD", "RAM", "LAL", "AHMED", "KHAN", "ANSARI", "SIDDIQUI",
        "ALI", "HUSSAIN", "RAJ", "SONI", "JAIN", "AGARWAL", "GOYAL", "MITTAL",
        "MALIK", "KAUR", "GILL", "DHILLON", "BRAR", "SANDHU", "STORE", "STORES",
        "MART", "TRADERS", "ENTERPRISES", "AGENCIES", "BROTHERS", "AND", "CO",
        "COMPANY", "INDUSTRIES", "PHARMA", "MEDICAL", "HOSPITAL", "CLINIC",
        "RESTAURANT", "HOTEL", "GARMENTS", "TEXTILES", "FASHION", "JEWELLERS",
        "FURNITURE", "ELECTRONICS", "AUTOMOBILES", "CONSTRUCTION", "BUILDER"
    }
    
    # ADDED: Garbage patterns to reject
    GARBAGE_PATTERNS = [
        r'^[OIZ]{5,}$',  # Only O,I,Z characters
        r'^[A-Z][OIZ]{4,}$',  # Letter followed by O/I/Z
        r'.*[OIZ]{5,}.*',  # Long O/I/Z sequences
        r'^[A-Z]{1,2}$',  # Very short
        r'^[A-Z][\s]*[A-Z]$',  # Two letters with space
    ]
    
    # Fixed OPR_NO value for all records
    FIXED_OPR_NO = "AS601"
    
    # Fixed FILE_MARK value for all records (Logical/Boolean - False)
    FIXED_FILE_MARK = False  # This will be stored as .F. in DBF
    
    # ADDED: Track statistics
    stats = {
        'processed': 0,
        'valid_names': 0,
        'xxx': 0,
        'garbage_rejected': 0
    }
    
    def is_handwritten(self, image):
        """Detect if text in the image is handwritten"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False
        
        handwritten_score = 0
        valid_contours = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            
            valid_contours += 1
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                handwritten_score += 1
            if extent < 0.4:
                handwritten_score += 1
            if solidity < 0.8:
                handwritten_score += 1
        
        if valid_contours == 0:
            return False
        
        avg_score = handwritten_score / valid_contours if valid_contours > 0 else 0
        
        if valid_contours > 15 and avg_score > 1.2:
            return True
        
        return False
    
    # MODIFIED: Better garbage detection
    def is_garbage(self, text):
        """Enhanced garbage detection"""
        if not text:
            return True
            
        letters = len(re.findall(r'[A-Za-z]', text))
        digits = len(re.findall(r'[0-9]', text))

        if letters < 3:
            return True

        if digits > letters:
            return True

        # Check for repeated O/I/Z (OCR errors)
        if re.search(r'O{4,}|I{4,}|Z{4,}', text):
            return True
            
        # Check garbage patterns
        for pattern in self.GARBAGE_PATTERNS:
            if re.match(pattern, text):
                return True

        return False
    
    # MODIFIED: Enhanced name scoring
    def score_name(self, text):
        """Better name scoring with common name detection"""
        score = 0
        words = text.split()
        text_upper = text.upper()

        if 2 <= len(words) <= 5:
            score += 0.3

        if all(word.isalpha() for word in words):
            score += 0.3
            
        # ADDED: Boost if contains common name parts
        common_matches = sum(1 for word in words if word.upper() in self.COMMON_NAME_PARTS)
        if common_matches > 0:
            score += 0.2 * common_matches

        return min(score, 1.0)  # Cap at 1.0
 
    def contains_bank_keyword(self, text):
        text_upper = text.upper()
        for word in self.BANK_TERMS:
            if word in text_upper:
                return True
        return False
     
    def __init__(self, args):
        self.input_dir = Path(args.input_dir)
        self.limit = args.limit
        self.dbf_path = Path(args.dbf_path)
        self.threshold = args.threshold
        self.output_dbf_path = None

        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            lang="en",
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
            writer.writerow(["image_name", "issuer_name", "opr_no", "file_mark", "confidence", "status"])

        if not self.dbf_path.exists():
            raise FileNotFoundError("DBF file not found.")

        self.table = dbf.Table(str(self.dbf_path))
        self.table.open(mode=dbf.READ_WRITE)

        self.drawer_dict = self.build_drawer_dict()
        print(f"Drawer dictionary loaded: {len(self.drawer_dict)} names")
        print(f"Fixed OPR_NO value for all records: {self.FIXED_OPR_NO}")
        print(f"Fixed FILE_MARK value for all records: {self.FIXED_FILE_MARK} (Logical/Boolean - .F.)")

    def clean_text(self, text):
        if not text:
            return "XXX"

        text = text.upper()
        # MODIFIED: More careful replacements
        text = text.replace("0", "O").replace("1", "I").replace("|", "I").replace("5", "S")
        # Don't replace all special chars - keep & and . for business names
        text = re.sub(r"[^A-Z0-9 .,&/-]", " ", text)
        text = " ".join(text.split())
        return text.strip()

    def is_junk_text(self, text):
        if not text:
            return True

        if len(text) < 3:
            return True

        text_upper = text.upper()

        # Direct junk match
        for term in self.JUNK_TERMS:
            if term in text_upper:
                return True

        # Fuzzy sign detection
        if "SIGN" in text_upper or "SGN" in text_upper:
            return True

        # Pay to order / bearer block
        if "BEARER" in text_upper or "ORDER" in text_upper:
            return True

        return False

    # MODIFIED: Enhanced validation
    def is_valid_drawer(self, text):
        if self.is_junk_text(text):
            return False
        if self.contains_bank_keyword(text):
            return False
        
        if "BANK" in text:
            return False

        if "PLEASE" in text:
            return False

        # MODIFIED: Allow dots in business names (like M/S)
        if text.count('.') > 2 and 'M/S' not in text and 'M/S.' not in text:
            return False
            
        location_keywords = [
            "ROAD", "NAGAR", "COLONY", "MUMBAI", "DELHI",
            "FARIDABAD", "STATE", "DIST", "PIN"
        ]

        if any(loc in text for loc in location_keywords):
            return False

        alpha = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        total = len(text)
        if total == 0:
            return False

        # Avoid mostly numeric non-name strings.
        if digits > alpha and alpha < 4:
            return False

        alpha_ratio = alpha / total
        if total > 6 and alpha_ratio < 0.45:
            return False

        bank_hits = sum(1 for term in self.BANK_TERMS if term in text)
        if bank_hits > 0 and alpha < 8:
            return False
        
        words = text.split()
        
        # MODIFIED: More lenient length for Indian names
        if len(text) < 5:  # Changed from 8 to 5
            return False

        # reject strings with too many digits
        digits = sum(c.isdigit() for c in text)
        letters = sum(c.isalpha() for c in text)

        if digits >= 3:
            return False

        # MODIFIED: Allow single word names if they look valid
        if " " not in text and digits == 0 and letters >= 5:
            # Single word name - check if it looks like a name
            if not self.is_garbage(text):
                pass  # Allow it
            else:
                return False
        elif " " not in text and digits > 0:
            return False

        # reject amount keywords
        if "NOTOVER" in text:
            return False

        if "NOT OVER" in text:
            return False

        if "RS" in text and len(words) <= 3:
            return False
            
        # MODIFIED: Allow / in M/S format
        if "/" in text and "M/S" not in text and "M/S." not in text:
            return False
            
        # MODIFIED: Allow - in hyphenated names
        if "-" in text and len(text.split('-')) > 3:
            return False
            
        # MODIFIED: Allow . in abbreviations
        if "." in text and not any(c.isalpha() for c in text.replace('.', '')):
            return False  
            
        # reject lines that contain mostly digits
        digits = sum(c.isdigit() for c in text)
        letters = sum(c.isalpha() for c in text)

        if digits > letters:
            return False

        # At least 60% letters (more lenient)
        letters = sum(c.isalpha() for c in text)
        if letters / len(text) < 0.6:
            return False

        # ADDED: Check for garbage patterns
        if self.is_garbage(text):
            self.stats['garbage_rejected'] += 1
            return False

        return True

    def build_drawer_dict(self):
        names = set()
        for record in list(self.table):
            drawer = self.clean_text(str(record.DRAWER_NM).strip())
            if self.is_valid_drawer(drawer):
                names.add(drawer)
        return sorted(names)

    def match_name(self, text):
        if not text:
            return None, 0.0

        text = text.upper()

        # Exact match first.
        if text in self.drawer_dict:
            return text, 1.0

        matches = difflib.get_close_matches(text, self.drawer_dict, n=1, cutoff=0.78)
        if not matches:
            return None, 0.0

        best = matches[0]
        similarity = difflib.SequenceMatcher(None, text, best).ratio()
        return best, similarity

    # MODIFIED: Better FOR pattern handling
    def extract_issuer(self, ocr_result):
        if not ocr_result:
            return "XXX", 0.0, -1.0
      
        candidates = []

        for line in ocr_result:
            if len(line) < 2:
                continue

            raw_text = line[1][0].strip()
            conf = float(line[1][1])

            text = self.clean_text(raw_text)
            
            # ADDED: Special handling for FOR patterns
            if "FOR " in text or "F/O " in text or "F/" in text or "PROP " in text:
                # Extract after FOR
                for marker in ["FOR ", "F/O ", "F/ ", "PROP "]:
                    if marker in text:
                        parts = text.split(marker, 1)
                        if len(parts) > 1:
                            potential_name = parts[1].strip()
                            if self.is_valid_drawer(potential_name):
                                score = conf + 0.3  # Boost FOR patterns
                                candidates.append((score, conf, potential_name))
                                break
                continue  # Skip normal processing for FOR lines
        
            if not self.is_valid_drawer(text):
                continue

            matched_name, similarity = self.match_name(text)

            # Decide final text first
            final_text = matched_name if matched_name else text

            # Initialize score safely
            score = conf

            # Boost for dictionary match
            if matched_name:
                score += 0.20 + (0.25 * similarity)

            # Boost for good word count
            words = final_text.split()
            if 2 <= len(words) <= 4:
                score += 0.15
                
            # ADDED: Boost for common name parts
            common_matches = sum(1 for word in words if word.upper() in self.COMMON_NAME_PARTS)
            if common_matches > 0:
                score += 0.1 * common_matches

            # Penalize suspicious words
            suspicious = ["PLEASE", "SIGN", "ABOVE", "BEARER", "ORDER", "CTS CLEARING", 
                         "SAVINGSAC", "PROPRIETOR", "CURRENTAC", "A/C", "PAYEE", "ONLY", 
                         "NOTOVER", "NOT OVER", "RS"]
            if any(s in final_text for s in suspicious):
                score -= 0.40

            candidates.append((score, conf, final_text))

        if not candidates:
            return "XXX", 0.0, -1.0

        candidates.sort(reverse=True, key=lambda x: x[0])
        best_score, best_conf, best_text = candidates[0]

        return best_text, best_conf, best_score
    
    # MODIFIED: Better name validation
    def process_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            return "XXX", 0.0
  
        h, w, _ = img.shape
  
        # Extract ROI for issuer name
        issuer_roi = img[int(h * 0.40):int(h * 0.90), int(w * 0.35):w]

        # Process issuer ROI
        issuer_result = self.ocr.ocr(issuer_roi, cls=True)

        # Extract issuer name
        if not issuer_result or not issuer_result[0]:
            return "XXX", 0.0
    
        # First check for "FOR <NAME>" pattern
        for line in issuer_result[0]:
            if len(line) > 1:
                text = line[1][0].strip()
                text_upper = text.upper()
                
                # Check all FOR-like patterns
                for marker in ["FOR ", "F/O ", "F/ ", "PROP "]:
                    if marker in text_upper:
                        issuer = text[text_upper.find(marker) + len(marker):].strip()
                        issuer_conf = float(line[1][1])
                        
                        # Validate the extracted name
                        if self.looks_like_real_name(issuer) and issuer_conf >= self.threshold:
                            return issuer, issuer_conf
        
        # If no FOR pattern, get best candidate from extract_issuer
        best_text, best_conf, best_score = self.extract_issuer(issuer_result[0])
    
        # STRICT VALIDATION
        if best_text == "XXX":
            return "XXX", 0.0
    
        if best_conf < self.threshold:
            return "XXX", 0.0
    
        if not self.looks_like_real_name(best_text):
            return "XXX", 0.0
    
        return best_text, best_conf

    # MODIFIED: Enhanced name validation
    def looks_like_real_name(self, text):
        """Enhanced validation - reject garbage, keep real names"""
        if not text or text == "XXX":
            return False
       
        # Count letters and digits
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        total = len(text)
    
        # If it's mostly digits, reject
        if digits > letters and letters < 5:
            return False
    
        # If it's too short and not common abbreviation
        if total < 4 and text.upper() not in ["CO", "LTD", "PVT", "INC", "M/S"]:
            return False
    
        # Reject if contains too many special characters
        special = sum(not c.isalnum() and c != ' ' for c in text)
        if special > 3:
            return False
    
        # ADDED: Check for garbage patterns
        if self.is_garbage(text):
            return False
    
        # Suspicious words check
        text_upper = text.upper()
        suspicious = ["BANK", "IFSC", "BRANCH", "PLEASE", "SIGN", 
                     "ACCOUNT", "AMOUNT", "RUPEES", "PAYEE",
                     "BEARER", "ORDER", "STAMP"]
    
        for s in suspicious:
            if s in text_upper and len(s) > len(text)/2:
                return False
    
        # ADDED: Must have at least one common name part or be reasonable length
        words = text_upper.split()
        common_matches = sum(1 for word in words if word in self.COMMON_NAME_PARTS)
        
        if common_matches == 0 and len(words) >= 2:
            # Check if all words look like names (alphabetic, no garbage)
            for word in words:
                if len(word) < 2 or self.is_garbage(word):
                    return False
        elif common_matches == 0 and len(words) == 1:
            # Single word must be reasonable length and not garbage
            if len(text) < 5 or self.is_garbage(text):
                return False
    
        return True
  
    def update_original_dbf(self, image_name, issuer):
        """Update the original source DBF"""
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
        """Append results to the output DBF"""
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
            p
            for p in self.input_dir.iterdir()
            if p.suffix.lower() in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
        )

        if self.limit:
            images = images[: self.limit]

        print(f"Processing {len(images)} images...")
        print(f"Fixed OPR_NO for all records: {self.FIXED_OPR_NO}")
        print(f"Fixed FILE_MARK for all records: {self.FIXED_FILE_MARK} (.F.)")
        
        batch_size = 200
        total = len(images)
        
        # Reset stats
        self.stats = {
            'processed': 0,
            'valid_names': 0,
            'xxx': 0,
            'garbage_rejected': 0
        }

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_images = images[start:end]

            print(f"\nProcessing batch {start + 1} to {end}...\n")

            # Restart OCR to refresh memory
            self.ocr = PaddleOCR(use_gpu=False, use_angle_cls=True, lang="en", show_log=False)

            handwritten_count = 0 
            for img_path in batch_images:
                
                try:
                    issuer, confidence = self.process_image(img_path)
                    
                    # Update statistics
                    self.stats['processed'] += 1
                    
                    if issuer == "XXX":
                        self.stats['xxx'] += 1
                        status = "REJECTED"
                    else:
                        self.stats['valid_names'] += 1
                        status = "AUTO-UPDATED"

                    # Append to output DBF
                    self.append_to_output_dbf(img_path.name, issuer, confidence, status)

                    # Always update original DBF
                    self.update_original_dbf(img_path.name, issuer)

                    with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([img_path.name, issuer, self.FIXED_OPR_NO, 
                                       self.FIXED_FILE_MARK, round(confidence, 4), status])

                    print(f"SUMMARY: {img_path.name} -> Issuer: {issuer}, "
                          f"OPR_NO: {self.FIXED_OPR_NO}, FILE_MARK: {self.FIXED_FILE_MARK} (.F.) "
                          f"({round(confidence, 2)}) [{status}]")

                except Exception as e:
                    print(f"Error processing {img_path.name}: {e}")
                    traceback.print_exc()

        # Close all tables
        self.output_table.close()
        self.table.close()
        
        # Print final statistics
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE!")
        print(f"{'='*50}")
        print(f"Output DBF file: {self.output_dbf_path}")
        print(f"File size: {self.output_dbf_path.stat().st_size} bytes")
        print(f"\nFINAL STATISTICS:")
        print(f"  Total processed: {self.stats['processed']}")
        print(f"  Valid names extracted: {self.stats['valid_names']}")
        print(f"  XXX returned: {self.stats['xxx']}")
        print(f"  Garbage rejected: {self.stats['garbage_rejected']}")
        if self.stats['processed'] > 0:
            accuracy = (self.stats['valid_names'] / self.stats['processed']) * 100
            print(f"  Accuracy: {accuracy:.2f}%")
        print(f"\nFixed OPR_NO value used for all records: {self.FIXED_OPR_NO}")
        print(f"Fixed FILE_MARK value used for all records: {self.FIXED_FILE_MARK} (.F.)")
        print(f"\nCOPY THIS FILE to your Visual FoxPro folder:")
        print(f"  {self.output_dbf_path}")
        print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--dbf-path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    processor = IssuerBatchProcessorV3(args)
    processor.run()