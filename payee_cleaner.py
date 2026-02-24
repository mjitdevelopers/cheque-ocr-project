#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Payee Name Cleaning Pipeline
Removes titles, replaces & with AND, applies XXX fallback
Last Updated: 2025
"""

import re
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class PayeeCleaner:
    """
    Standardized payee name cleaning:
    1. Remove honorifics/titles (Dr, Adv, M/s, etc.)
    2. Replace & with AND
    3. Remove company suffixes
    4. Trim and normalize whitespace
    5. Apply XXX fallback
    """
    
    def __init__(self):
        # Titles to remove (comprehensive Indian context)
        self.titles = [
            # Professional titles
            r'^DR\.?\s+', r'^DR\s+',                    # Doctor
            r'^ADV\.?\s+', r'^ADV\s+',                  # Advocate
            r'^CA\.?\s+', r'^CA\s+',                    # Chartered Accountant
            r'^CS\.?\s+', r'^CS\s+',                    # Company Secretary
            r'^CMA\.?\s+', r'^CMA\s+',                  # Cost Accountant
            r'^ICWA\.?\s+', r'^ICWA\s+',                # Cost Accountant (old)
            r'^ENG\.?\s+', r'^ENG\s+',                  # Engineer
            r'^ARCH\.?\s+', r'^ARCH\s+',                # Architect
            r'^PROF\.?\s+', r'^PROF\s+',                # Professor
            
            # Honorifics
            r'^MR\.?\s+', r'^MR\s+',                    # Mister
            r'^MRS\.?\s+', r'^MRS\s+',                  # Missus
            r'^MS\.?\s+', r'^MS\s+',                    # Miss
            r'^MISS\.?\s+', r'^MISS\s+',                # Miss
            r'^MASTER\.?\s+', r'^MASTER\s+',            # Master (young male)
            r'^KUM\.?\s+', r'^KUMARI\.?\s+',            # Kumari (unmarried woman)
            r'^SMT\.?\s+', r'^SMT\s+',                  # Smt (married woman)
            r'^SHRI\.?\s+', r'^SHRI\s+',                # Shri (Mr)
            r'^SHRIMATI\.?\s+', r'^SHRIMATI\s+',        # Shrimati (Mrs)
            
            # Business entities
            r'^M/S\.?\s+', r'^M/S\s+', r'^M/S[.]?\s*',  # Proprietorship
            r'^MESSRS\.?\s+', r'^MESSRS\s+',            # Messrs
            r'^MSME\.?\s+', r'^MSME\s+',                # Micro/Small Enterprise
            
            # Religious titles
            r'^PT\.?\s+', r'^PANDIT\.?\s+',            # Pandit
            r'^SWAMI\.?\s+',                            # Swami
            r'^MAULVI\.?\s+',                           # Maulvi
            r'^MAULANA\.?\s+',                          # Maulana
            r'^QAZI\.?\s+',                             # Qazi
            r'^FATHER\.?\s+', r'^FR\.?\s+',             # Father
            r'^BROTHER\.?\s+', r'^BR\.?\s+',            # Brother
            r'^SISTER\.?\s+', r'^SR\.?\s+',             # Sister
        ]
        
        # Company suffixes to remove
        self.company_suffixes = [
            r'\s+PVT\.?\s+LTD\.?$', r'\s+PVT\s+LTD\.?$',
            r'\s+PRIVATE\s+LIMITED$',
            r'\s+LTD\.?$', r'\s+LIMITED$',
            r'\s+LLP$', r'\s+LLC$',
            r'\s+INC\.?$', r'\s+INCORPORATED$',
            r'\s+CO\.?$', r'\s+COMPANY$',
            r'\s+CORP\.?$', r'\s+CORPORATION$',
            r'\s+PVT$', r'\s+PRIVATE$',
            r'\s+ENTERPRISES$', r'\s+ENTERPRISE$',
            r'\s+INDUSTRIES$', r'\s+INDUSTRY$',
            r'\s+TRADERS$', r'\s+TRADING$',
            r'\s+CONTRACTORS$', r'\s+CONTRACTOR$',
            r'\s+SUPPLIERS$', r'\s+SUPPLIER$',
            r'\s+WORKS$', r'\s+WORK$',
            r'\s+ASSOCIATES$', r'\s+ASSOCIATE$',
            r'\s+GROUP$', r'\s+HOLDINGS$',
        ]
        
        # Special characters to clean
        self.special_chars = {
            '&': 'AND',
            '＆': 'AND',  # Full-width ampersand
            '+': 'AND',
            '@': 'AT',
            '%': 'PERCENT',
            '#': 'NUMBER',
            '$': 'DOLLAR',
            '€': 'EURO',
            '£': 'POUND',
            '¥': 'YEN',
        }
    
    def clean(self, raw_payee: Optional[str]) -> str:
        """
        Complete payee cleaning pipeline
        Returns cleaned name or "XXX" if empty
        """
        if not raw_payee:
            logger.debug("Empty payee, returning XXX")
            return "XXX"
        
        # Convert to uppercase for consistent processing
        text = raw_payee.upper().strip()
        
        # Remove leading titles
        original = text
        for pattern in self.titles:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        if text != original:
            logger.debug(f"Removed title: '{original}' -> '{text}'")
        
        # Replace special characters
        for char, replacement in self.special_chars.items():
            if char in text:
                text = text.replace(char, replacement)
                logger.debug(f"Replaced '{char}' with '{replacement}'")
        
        # Remove trailing company suffixes
        original = text
        for pattern in self.company_suffixes:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        if text != original:
            logger.debug(f"Removed company suffix: '{original}' -> '{text}'")
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\-]', '', text)  # Keep letters, numbers, spaces, dots, hyphens
        
        # Apply XXX rule if empty or too short
        if not text or len(text) < 2:
            logger.debug(f"Payee too short '{text}', returning XXX")
            return "XXX"
        
        logger.debug(f"Final cleaned payee: '{text}'")
        return text
    
    def extract_payee_from_text(self, text: str) -> Optional[str]:
        """
        Extract payee name from full OCR text using cheque patterns
        """
        if not text:
            return None
        
        text_upper = text.upper()
        
        # Payee patterns in order of reliability
        patterns = [
            # Standard patterns
            (r'PAY\s*TO\s*[:\s]*([A-Z][A-Z\s\.]+?)(?=\s+RUPEES|\s+RS\.?|\s*$)', 1),
            (r'ORDER\s*OF\s*[:\s]*([A-Z][A-Z\s\.]+?)(?=\s+RUPEES|\s+RS\.?|\s*$)', 1),
            (r'[Pp][Aa][Yy]\s*[:\s]*([A-Z][A-Z\s\.]{2,30})', 1),
            (r'BEARER\s*[:\s]*([A-Z][A-Z\s\.]+)', 1),
            (r'OR\s*BEARER\s*[:\s]*([A-Z][A-Z\s\.]+)', 1),
            
            # Alternative patterns
            (r'FAVOUR\s+OF\s*[:\s]*([A-Z][A-Z\s\.]+)', 1),
            (r'FAVOR\s+OF\s*[:\s]*([A-Z][A-Z\s\.]+)', 1),
            (r'BENEFICIARY\s*[:\s]*([A-Z][A-Z\s\.]+)', 1),
            
            # Fallback: name before RUPEES
            (r'([A-Z][A-Z\s\.]{3,30})\s+(?:RUPEES|RS\.?)', 1),
            
            # Last resort: first capitalized line that's not a bank name
            (r'^([A-Z][A-Z\s\.]{5,50})$', 1),
        ]
        
        for pattern, group_idx in patterns:
            match = re.search(pattern, text_upper)
            if match:
                candidate = match.group(group_idx).strip()
                # Filter out bank names and common non-payee text
                if not self._is_bank_or_metadata(candidate):
                    return candidate
        
        return None
    def extract_all_fields(self, text):
      """Extract ALL underlined fields from OCR text"""
      fields = {}
    
     # Extract date
      date_match = re.search(r'Date:\s*(\d{2})[./](\d{2})[./](\d{4})', text)
      if date_match:
        fields['date'] = f"{date_match.group(1)}{date_match.group(2)}{date_match.group(3)}"
    
     # Extract amount
      amount_match = re.search(r'[₹Rs\.]*\s*([\d,]+(?:\.\d{2})?)[/\-]?', text)
      if amount_match:
        fields['amount'] = amount_match.group(1).replace(',', '')
    
     # Extract account
      account_match = re.search(r'A/C No\.?[\s:]*(\d{10,20})', text)
      if account_match:
        fields['account'] = account_match.group(1)
    
     # Extract payee
      payee_match = re.search(r'PAY\s+([A-Z\s]+?)(?=\s+RUPEE|\s+Rs|\s+₹|$)', text)
      if payee_match:
        fields['payee'] = payee_match.group(1).strip()
    
      return fields

    def _is_bank_or_metadata(self, text: str) -> bool:
        """Check if text is bank name or metadata rather than payee"""
        text_upper = text.upper()
        
        # Bank identifiers
        banks = ['STATE BANK', 'SBI', 'BANK OF', 'CANARA', 'PNB',
                'BOB', 'HDFC', 'ICICI', 'AXIS', 'KOTAK', 'YES BANK',
                'IDBI', 'UNION BANK', 'INDIAN BANK', 'CENTRAL BANK']
        
        for bank in banks:
            if bank in text_upper:
                return True
        
        # Metadata keywords
        metadata = ['BRANCH', 'IFSC', 'MICR', 'CODE', 'CITY', 'DELHI',
                   'MUMBAI', 'KOLKATA', 'CHENNAI', 'BANGALORE']
        
        for word in metadata:
            if word in text_upper:
                return True
        
        return False


# Singleton instance
_cleaner_instance = None

def get_cleaner() -> PayeeCleaner:
    """Get or create cleaner singleton"""
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = PayeeCleaner()
    return _cleaner_instance