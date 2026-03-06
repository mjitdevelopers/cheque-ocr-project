import os
import csv
import argparse
from pathlib import Path
from datetime import datetime
import re
import difflib
import cv2
from paddleocr import PaddleOCR
import dbf

class IssuerBatchProcessorV3:
    def __init__(self, args):
        self.input_dir = Path(args.input_dir)
        self.limit = args.limit
        self.dbf_path = Path(args.dbf_path)
        self.threshold = args.threshold
        self.debug = args.debug

        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            use_gpu=args.use_gpu,
            use_angle_cls=True,
            lang="en",
            show_log=args.debug
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("issuer_output")
        output_dir.mkdir(exist_ok=True)

        self.csv_path = output_dir / f"ISSUER_RESULTS_{timestamp}.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "issuer_name", "confidence", "status", "matched_with"])

        print(f"CSV Created: {self.csv_path}")

        if not self.dbf_path.exists():
            raise FileNotFoundError(f"DBF file not found: {self.dbf_path}")

        self.table = dbf.Table(str(self.dbf_path))
        self.table.open(mode=dbf.READ_WRITE)
        print("DBF Opened Successfully.")

        # Build drawer name dictionary from DBF
        self.drawer_dict = self.build_drawer_dict()
        print(f"Drawer dictionary loaded: {len(self.drawer_dict)} names")
        
        # Cache for matched names
        self.match_cache = {}

    # Build drawer dictionary from DBF
    def build_drawer_dict(self):
        names = set()
        for record in self.table:
            drawer = str(record.DRAWER_NM).strip()
            if drawer and len(drawer) > 2:
                drawer = drawer.upper()
                if self.is_valid_business_name(drawer):
                    names.add(drawer)
        return sorted(list(names))

    # Comprehensive list of junk words and patterns
    JUNK_PATTERNS = [
        # Signature related
        r'AUTHORISED?S?I?Q?N?A?T?O?R?Y?',  # Matches various misspellings of AUTHORISED/SIGNATORY
        r'AUTHORIZEDS?I?Q?N?A?T?O?R?Y?',
        r'SIGNAT(?:ORY|URE|ORY\(IES\))',
        r'PLEASE\s+SIGN\s+ABOVE',
        r'PLEASE\s+SGN\s+ABOVE',
        r'SIGN\s+ABOVE',
        r'SIGN\s+HERE',
        r'SIGN\s+BELOW',
        
        # Proprietor/Partner related
        r'PROPRIETOR',
        r'PROPRIER',
        r'PROP\.',
        r'PARTNERS?',
        r'PARTNERSHIP',
        
        # Account/Cheque related
        r'CURRENT',
        r'SAVINGS',
        r'CASH\s+CREDIT',
        r'OVERDRAFT',
        r'LOAN\s+ACCOUNT',
        r'ACCOUNT\s+PAYEE',
        r'A/C\s+PAYEE',
        r'PAYEE',
        r'DRAWER',
        r'DRAWEE',
        
        # Generic bank related
        r'BANK\s+COPY',
        r'BANK\s+USE\s+ONLY',
        r'FOR\s+BANK\s+USE',
        r'BRANCH\s+COPY',
        r'CUSTOMER\s+COPY',
        
        # Transaction related
        r'CHEQUE',
        r'CHECK',
        r'PAY\s+TO',
        r'PAYABLE',
        r'AMOUNT',
        r'RUPEES',
        r'RS\.',
        r'TOTAL',
        r'BALANCE',
        
        # Other common junk
        r'EXECUTOR',
        r'TRUSTEE',
        r'GUARDIAN',
        r'CANCELLED',
        r'VOID',
        r'SAMPLE',
        r'SPECIMEN',
        r'DUPLICATE',
        r'COUNTERFOIL',
        r'STUB',
        r'CHALLAN',
        r'FORM',
        
        # Single letter combinations (often misreads)
        r'^[A-Z]{2,4}$',  # 2-4 letter all-caps words like JIQ, VFG, JDP
    ]

    def is_junk_text(self, text):
        """Check if text is common cheque footer junk"""
        if not text or len(text) < 2:
            return True
        
        text_upper = text.upper()
        
        # Check for very short all-caps words (likely not business names)
        if re.match(r'^[A-Z]{2,4}$', text_upper) and text_upper not in ['LTD', 'PVT', 'INC', 'CORP', 'CO.', 'M/S']:
            return True
        
        # Check for patterns
        for pattern in self.JUNK_PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                # If the matched text is a significant portion of the original
                match = re.search(pattern, text_upper, re.IGNORECASE)
                if match and len(match.group()) >= len(text_upper) * 0.6:
                    return True
        
        # Check for common OCR errors of junk text
        junk_variations = [
            'AUTHORISED', 'AUTHORIZED', 'AUTHORIS', 'AUTHORIZ',
            'SIGNATORY', 'SIGNATURE', 'SIGN', 'SGN',
            'PROPRIETOR', 'PROPRIER', 'PROP',
            'CURRENT', 'SAVINGS', 'ACCOUNT',
            'PAYEE', 'DRAWER', 'CHEQUE', 'CHECK',
            'BANK', 'BRANCH', 'COPY',
            'AMOUNT', 'RUPEES', 'TOTAL', 'BALANCE'
        ]
        
        for junk in junk_variations:
            if junk in text_upper:
                # If the junk word makes up most of the text, it's junk
                if len(junk) >= len(text_upper) * 0.5:
                    return True
        
        return False

    # Clean OCR text
    def clean_text(self, text):
        if not text:
            return ""
        
        # Fix common OCR misreads
        text = text.upper()
        
        # OCR confusion mappings
        ocr_mappings = {
            '|': 'I',
            '!': 'I',
            'l': 'I',
            '0': 'O',
            '1': 'I',
            '5': 'S',  # Sometimes 5 is misread as S
            '8': 'B',  # Sometimes 8 is misread as B
        }
        
        # Apply mappings
        for wrong, correct in ocr_mappings.items():
            text = text.replace(wrong, correct)
        
        # Remove unwanted characters
        text = re.sub(r"[^A-Za-z0-9 ,.&/-]", " ", text)
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text.strip()

    # Check if text looks like a valid business name
    def is_valid_business_name(self, text):
        if not text or len(text) < 3:
            return False
        
        # First check if it's junk
        if self.is_junk_text(text):
            return False
        
        # Skip pure bank names
        bank_names = [
            'BANK', 'STATE BANK', 'HDFC', 'ICICI', 'SBI', 'AXIS', 
            'YES BANK', 'IDFC', 'KOTAK', 'INDUSIND', 'PNB', 'CANARA',
            'UNION BANK', 'BOB', 'BARODA', 'SYNDICATE', 'CORPORATION',
            'BANK OF INDIA', 'BANK OF BARODA', 'PUNJAB NATIONAL BANK'
        ]
        
        text_upper = text.upper()
        for bank in bank_names:
            if bank in text_upper:
                # If it's mostly the bank name, reject
                if len(bank) >= len(text_upper) * 0.7:
                    return False
        
        # Check composition
        alpha_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        
        # Business names should have at least some alphabetic characters
        if digit_count > 0 and alpha_count == 0:
            # Pure numbers - only accept if 8+ digits (account numbers)
            return len(text) >= 8
        
        # For mixed content, at least 30% should be alphabetic
        total_chars = len(text)
        if alpha_count > 0:
            alpha_ratio = alpha_count / total_chars
            if alpha_ratio < 0.3 and total_chars > 5:
                # Too many non-alpha characters for a business name
                # But could be a code
                return digit_count > 5  # Accept if it's mostly digits (code)
        
        return True

    # Validate if text is suitable for drawer name
    def is_valid_drawer(self, text):
        if not text:
            return False
        
        # Remove very short texts
        if len(text) < 3:
            return False
        
        # Check if it's junk
        if self.is_junk_text(text):
            return False
        
        return self.is_valid_business_name(text)

    # Remove common prefixes
    def remove_prefixes(self, text):
        """Remove common prefixes like 'FOR', 'M/S', etc."""
        if not text:
            return text
        
        # Common patterns to remove
        patterns = [
            r'^FOR\s+',
            r'^FOR\s+M/S\s+',
            r'^M/S\s+',
            r'^M/S\.\s+',
            r'^MESSRS\s+',
            r'^MESSRS\.\s+',
            r'^SHRI\s+',
            r'^SRI\s+',
            r'^MR\.\s+',
            r'^MRS\.\s+',
            r'^MS\.\s+',
            r'^DR\.\s+',
        ]
        
        cleaned = text.upper()
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

    # Fuzzy match OCR text with DBF drawer names
    def match_name(self, text):
        if not text:
            return None
        
        # Check cache first
        cache_key = text.upper()
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]
        
        # Remove prefixes before matching
        text_without_prefix = self.remove_prefixes(text)
        
        # Try exact match first
        for drawer in self.drawer_dict:
            if text.upper() == drawer or text_without_prefix.upper() == drawer:
                self.match_cache[cache_key] = drawer
                return drawer
        
        # Try fuzzy match with different cutoffs
        for candidate_text in [text, text_without_prefix]:
            candidate_upper = candidate_text.upper()
            for cutoff in [0.85, 0.80, 0.75]:
                matches = difflib.get_close_matches(candidate_upper, self.drawer_dict, n=1, cutoff=cutoff)
                if matches:
                    self.match_cache[cache_key] = matches[0]
                    return matches[0]
        
        self.match_cache[cache_key] = None
        return None

    # Extract issuer from OCR result
    def extract_issuer(self, ocr_result):
        if not ocr_result or not ocr_result[0]:
            return "NO_VALID_TEXT", 0.0, None

        candidates = []
        for line in ocr_result[0]:
            txt = line[1][0].strip()
            conf = float(line[1][1])
            
            cleaned = self.clean_text(txt)
            
            # Skip if it's junk text
            if not cleaned or self.is_junk_text(cleaned):
                if self.debug:
                    print(f"    Filtered junk: '{cleaned}'")
                continue
            
            if self.is_valid_drawer(cleaned):
                candidates.append((conf, cleaned, txt))
            elif self.debug:
                print(f"    Invalid drawer: '{cleaned}'")

        if not candidates:
            return "NO_VALID_TEXT", 0.0, None

        # Sort by confidence
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Try to match with DBF
        best_conf, best_cleaned, best_original = candidates[0]
        
        matched_name = self.match_name(best_cleaned)
        if matched_name:
            return matched_name, best_conf, matched_name
        else:
            # Remove prefixes if present
            final_name = self.remove_prefixes(best_cleaned)
            return final_name, best_conf, None

    # Process single image
    def process_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            return "NO_VALID_TEXT", 0.0, None

        h, w, _ = img.shape
        
        # Try multiple crop regions
        crop_regions = [
            (int(h*0.45), h, int(w*0.35), w),  # Bottom-right
            (int(h*0.40), h, int(w*0.30), w),  # Slightly larger
            (int(h*0.30), h, int(w*0.20), w),  # Even larger
            (0, h, 0, w)  # Full image as last resort
        ]
        
        best_issuer = "NO_VALID_TEXT"
        best_confidence = 0.0
        best_match = None
        
        for i, (y1, y2, x1, x2) in enumerate(crop_regions):
            if self.debug:
                print(f"  Trying crop region {i+1}")
            cropped = img[y1:y2, x1:x2]
            result = self.ocr.ocr(cropped, cls=True)
            issuer, confidence, matched = self.extract_issuer(result)
            
            if confidence > best_confidence and issuer != "NO_VALID_TEXT":
                best_confidence = confidence
                best_issuer = issuer
                best_match = matched
                
                if confidence >= 0.95:  # High confidence, stop searching
                    break
        
        return best_issuer, best_confidence, best_match

    # Update DBF
    def update_dbf(self, image_name, issuer):
        image_base = Path(image_name).stem.strip().lower()
        updated = False
        
        for record in self.table:
            dbf_image = str(record["IMAGE_FILE"]).strip().lower()
            dbf_base = Path(dbf_image).stem.strip().lower()
            if dbf_base == image_base:
                with record:
                    old_value = str(record.DRAWER_NM).strip()
                    record.DRAWER_NM = issuer[:50]
                    if self.debug:
                        print(f"  Updated DBF: '{old_value}' -> '{issuer[:50]}'")
                updated = True
                break
        
        return updated

    # Run processor
    def run(self):
        if not self.input_dir.exists():
            print(f"Input directory not found: {self.input_dir}")
            return

        images = sorted([
            p for p in self.input_dir.iterdir()
            if p.suffix.lower() in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
        ])
        if self.limit:
            images = images[:self.limit]

        print(f"\nProcessing {len(images)} images...\n")
        
        stats = {
            'auto_updated': 0,
            'review': 0,
            'review_short': 0,
            'junk_filtered': 0,
            'invalid': 0,
            'no_text': 0,
            'error': 0
        }

        for img_path in images:
            try:
                if self.debug:
                    print(f"\nProcessing: {img_path.name}")
                    
                issuer, confidence, matched = self.process_image(img_path)
                
                # Determine status
                if issuer != "NO_VALID_TEXT" and self.is_junk_text(issuer):
                    status = "JUNK-FILTERED"
                    stats['junk_filtered'] += 1
                elif confidence >= self.threshold and issuer and issuer != "NO_VALID_TEXT":
                    if self.is_valid_drawer(issuer):
                        # Check if it's a questionable short name
                        if len(issuer) <= 4 and issuer.isalpha() and issuer not in ["LTD", "PVT", "INC", "CO."]:
                            status = "REVIEW-SHORT"
                            stats['review_short'] += 1
                        else:
                            updated = self.update_dbf(img_path.name, issuer)
                            if updated:
                                status = "AUTO-UPDATED"
                                stats['auto_updated'] += 1
                            else:
                                status = "MATCHED-NO-UPDATE"
                                stats['review'] += 1
                    else:
                        status = "INVALID-TEXT"
                        stats['invalid'] += 1
                elif issuer and issuer != "NO_VALID_TEXT" and self.is_valid_drawer(issuer):
                    status = "REVIEW"
                    stats['review'] += 1
                else:
                    issuer = "NO_VALID_TEXT"
                    status = "NO-TEXT"
                    stats['no_text'] += 1

                # Write to CSV
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        img_path.name, 
                        issuer, 
                        round(confidence, 4), 
                        status,
                        matched if matched else ""
                    ])

                print(f"{img_path.name} -> {issuer} ({round(confidence, 2)}) [{status}]")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                stats['error'] += 1
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([img_path.name, "ERROR", 0.0, f"ERROR: {str(e)}", ""])

        self.table.close()
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total images: {len(images)}")
        print(f"Auto-updated: {stats['auto_updated']}")
        print(f"Review needed: {stats['review']}")
        print(f"Review (short): {stats['review_short']}")
        print(f"Junk filtered: {stats['junk_filtered']}")
        print(f"Invalid text: {stats['invalid']}")
        print(f"No text found: {stats['no_text']}")
        print(f"Errors: {stats['error']}")
        print("="*50)
        print(f"\nResults saved to: {self.csv_path}")
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process issuer names from check images")
    parser.add_argument("--input-dir", required=True, help="Directory containing images")
    parser.add_argument("--dbf-path", required=True, help="Path to DBF file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold for auto-update")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()
    processor = IssuerBatchProcessorV3(args)
    processor.run()