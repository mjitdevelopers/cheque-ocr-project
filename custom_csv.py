# custom_csv.py - Creates CSV in YOUR desired format

import csv
import re
from pathlib import Path
from datetime import datetime

class CustomCSVWriter:
    """
    Creates CSV in your exact format:
    Payee Name, Date, Account Number, Reference, Sort Code, Amount, Filename
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.output_dir / f"cheque_output_{timestamp}.csv"
        
        # Create file with headers (optional - remove if you don't want headers)
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # writer.writerow(['Payee Name', 'Date', 'Account Number', 'Reference', 'Sort Code', 'Amount', 'Filename'])
    
    def extract_amount_from_filename(self, filename):
        """Try to extract amount from filename if present"""
        # Look for pattern like 159899.00 in filename
        match = re.search(r'(\d+\.\d{2})', filename)
        if match:
            return match.group(1)
        return "0.00"
    
    def extract_account_from_filename(self, filename):
        """Try to extract account number from filename"""
        # Look for numbers in filename
        numbers = re.findall(r'\d+', filename)
        if len(numbers) >= 1:
            return numbers[0]  # First number found
        return "0"
    
    def extract_date_from_filename(self, filename):
        """Try to extract date from filename (DDMMYYYY format)"""
        # Look for 8-digit number that could be a date
        matches = re.findall(r'(\d{8})', filename)
        if matches:
            return matches[0]
        return datetime.now().strftime('%d%m%Y')
    
    def write_result(self, ocr_result):
        """
        Write ONE result in your format:
        Payee Name, Date, Account Number, Reference, Sort Code, Amount, Filename
        """
        
        # Get values from OCR result
        payee_name = ocr_result.get('payee_cleaned', 'XXX')
        filename = Path(ocr_result.get('image_path', '')).name
        
        # Extract other fields from filename or set defaults
        date = self.extract_date_from_filename(filename)
        account = self.extract_account_from_filename(filename)
        reference = "0"  # You'll need to get this from somewhere
        sort_code = "0"  # You'll need to get this from somewhere
        amount = self.extract_amount_from_filename(filename)
        
        # Your exact format: Payee, Date, Account, Reference, Sort Code, Amount, Filename
        row = [
            payee_name,
            date,
            account,
            reference,
            sort_code,
            amount,
            filename
        ]
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        return row
    
    def write_batch(self, ocr_results):
        """Write multiple results at once"""
        rows = []
        for result in ocr_results:
            if result.get('success'):  # Only write successful ones
                row = self.write_result(result)
                rows.append(row)
        
        print(f"✅ Wrote {len(rows)} results to {self.csv_path}")
        return rows