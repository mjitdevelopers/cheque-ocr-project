# custom_csv.py - FINAL WORKING VERSION

import csv
import re
from pathlib import Path
from datetime import datetime

class CustomCSVWriter:
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # CRITICAL: This MUST be the filename
        self.csv_path = self.output_dir / f"PAYEE_NAMES_{timestamp}.csv"
        
        self.processed_files = set()
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Payee Name', 'Filename'])
        
        print(f"✅ CREATED: {self.csv_path}")
        print(f"📁 Look for this file: PAYEE_NAMES_{timestamp}.csv")
    
    def correct_spelling(self, name):
      """Fix ALL common OCR mistakes"""
    
    # Common OCR letter confusions
      char_fixes = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'J',
        'Q': 'O', 'B': 'R', 'D': 'O', 'P': 'R',
      }
    
    # Specific word corrections from your output
      word_corrections = {
         'STEELNDUSTRIES': 'STEEL INDUSTRIES',
        'STEELINDUSTRIES': 'STEEL INDUSTRIES',
        'STEELHDUSTRIES': 'STEEL INDUSTRIES',
        'STEELIHDUSTRIES': 'STEEL INDUSTRIES',
        'STEELIINDUSTRIES': 'STEEL INDUSTRIES',
        'RQR': '',
        'QRBEARER': '',
        'BEARER': '',
        'OR BEARER': '',
        'COAL': 'GOAL',
        'BALQJI': 'BALAJI',
        'ENTENPES': 'ENTERPRISES',
        'ENTSEPS': 'ENTERPRISES',
        'KISPA': 'KIXPA',
        'KIXPA': 'KIXPA',
        'AU': 'AU',
        'ARIHANTINEOCOO': 'ARIHANT NEOCO',
        'CURUGRAM': 'GURUGRAM',
        'RORBEARER': '',
        'RUPEESFFY': '',
        'FFFY': 'FIFTY',
        'FIRETHOTSONO': 'THOMSON',
        'THOLSONO': 'THOMSON',
        'OMSESFVRSID': '',
      }
    
      name = name.upper()
    
    # Apply character fixes first
      fixed_name = []
      for char in name:
        if char in char_fixes:
            fixed_name.append(char_fixes[char])
        else:
            fixed_name.append(char)
        name = ''.join(fixed_name)
    
    # Apply word corrections
      for wrong, correct in word_corrections.items():
        if wrong in name:
            name = name.replace(wrong, correct)
    
    # Clean up
      name = ' '.join(name.split())
    
    # If too short or garbage, mark as XXX
      if len(name) < 3 or any(garbage in name for garbage in ['RUPEES', 'WOTFTHO']):
        return "XXX"
    
      return name

    def write_result(self, ocr_result):
        """Write payee name"""
        
        filename = Path(ocr_result.get('image_path', '')).name
        
        if filename in self.processed_files:
            return False
        self.processed_files.add(filename)
        
        # Get payee from OCR result
        payee = ocr_result.get('payee_cleaned', 'XXX')
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([payee, filename])
        
        print(f"  ✓ {payee} - {filename}")
        return True
      
      # Add this new class to your existing custom_csv.py file
# Keep your existing CustomCSVWriter class untouched

class IssuerCSVWriter:
    """Extract issuer name from cheque and save to separate file"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.output_dir / f"ISSUER_NAMES_{timestamp}.csv"
        
        self.processed_files = set()
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Issuer Name', 'Filename'])
        
        print(f"✅ Issuer CSV created: {self.csv_path}")
    
    def extract_issuer_from_text(self, text):
        """Extract issuer name from cheque text"""
        if not text:
            return None
        
        text = text.upper()
        
        # Look for issuer/signatory patterns
        patterns = [
            r'FOR\s+([A-Z][A-Z\s\.]{3,50}?)(?:\s+Authorised|\s+Signature|\s*$)',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+Auth\s*Signatory',
            r'([A-Z][A-Z\s\.]{3,50}?)\s+Authorised\s+Signatory',
            r'FOR\s+([A-Z][A-Z\s\.]{3,50}?)(?:\s*$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                if len(name) > 3:
                    return name
        
        return None
    
    def write_result(self, ocr_result):
        """Write issuer name to CSV"""
        
        filename = Path(ocr_result.get('image_path', '')).name
        
        if filename in self.processed_files:
            return False
        self.processed_files.add(filename)
        
        text = ocr_result.get('full_text', '')
        issuer = self.extract_issuer_from_text(text) or "UNKNOWN"
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([issuer, filename])
        
        print(f"  👤 Issuer: {issuer} - {filename}")
        return True