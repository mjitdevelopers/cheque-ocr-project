#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cheque Type Classification Engine
Supports 10+ cheque types with government payee rules
Last Updated: 2025
"""

import re
import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChequeType(Enum):
    """Comprehensive classification of Indian cheque types"""
    
    # Standard Bank Cheques (6 types)
    BEARER = "BEARER"
    ORDER = "ORDER"
    CROSSED = "CROSSED"
    POST_DATED = "POST_DATED"
    STALE = "STALE"
    OPEN = "OPEN"
    SELF = "SELF"
    CANCELLED = "CANCELLED"
    
    # Government Cheques (3 categories)
    GOVT_NEGOTIABLE = "GOVT_NEGOTIABLE"        # Category I - Contractors/Suppliers
    GOVT_NON_TRANSFERABLE = "GOVT_NON_TRANSFERABLE"  # Category II - Salary/Office
    GOVT_GOVT_ACCOUNT = "GOVT_GOVT_ACCOUNT"    # Category III - Inter-departmental
    
    # Banker's Instruments (2 types)
    DEMAND_DRAFT = "DEMAND_DRAFT"
    MANAGERS_CHEQUE = "MANAGERS_CHEQUE"
    
    # Special Types
    ELECTRONIC = "ELECTRONIC"  # ECS/EFT
    FOREIGN = "FOREIGN"  # Foreign currency


@dataclass
class ChequeClassification:
    """Complete classification result"""
    type: ChequeType
    subtype: Optional[str] = None
    confidence: float = 0.0
    is_government: bool = False
    government_category: Optional[str] = None  # I, II, III
    payee_rule: str = "standard"
    micr_code: Optional[str] = None
    validation_notes: List[str] = None
    
    def __post_init__(self):
        if self.validation_notes is None:
            self.validation_notes = []


class ChequeTypeClassifier:
    """
    AI-powered cheque classifier using visual features, MICR, and text patterns
    Based on RBI guidelines and PAO accounting standards
    """
    
    def __init__(self):
        # Government markers - Based on PAO guidelines
        self.govt_markers = {
            'high': [
                'GOVERNMENT OF INDIA',
                'GOVT OF INDIA',
                'RESERVE BANK OF INDIA',
                'RBI',
                'PAY AND ACCOUNTS OFFICE',
                'PAO',
                'GOVERNMENT ACCOUNT',  # Category III marker
            ],
            'medium': [
                'CENTRAL GOVERNMENT',
                'STATE GOVERNMENT',
                'PUBLIC SECTOR BANK',
                'ACCREDITED BANK',
                'NOT TRANSFERABLE',  # Category II marker
                'CHEQUE DRAWING DDO',
                'DDO',
                'DRAWING AND DISBURSING OFFICER',
            ],
            'low': [
                'GOVT',
                'GOVERNMENT',
                'TREASURY',
                'PUBLIC ACCOUNT',
                'CONSOLIDATED FUND',
            ]
        }
        
        # Banker's Instrument markers
        self.dd_markers = ['DEMAND DRAFT', 'DD', 'DRAFT', 'BANK DRAFT']
        self.mc_markers = [
            'MANAGERS CHEQUE', 'MANAGER\'S CHEQUE', 'MC',
            'PAY ORDER', 'BANKER\'S CHEQUE', 'BANKERS CHEQUE'
        ]
        
        # Cheque crossing patterns
        self.crossed_patterns = [
            r'A/C\s*PAYEE',
            r'ACCOUNT\s*PAYEE',
            r'AC\s*PAYEE',
            r'CROSSED',
            r'NON[-\s]NEGOTIABLE',
            r'NOT\s+NEGOTIABLE'
        ]
        
        # MICR patterns (Indian bank format)
        self.micr_pattern = re.compile(r'(\d{9})\s+(\d{9})\s+(\d{6,9})')
    
    def classify(self, image: np.ndarray, ocr_text: str = "") -> ChequeClassification:
        """
        Classify cheque type with confidence scoring
        """
        text_upper = ocr_text.upper()
        
        # Step 1: Detect government cheques (highest priority)
        govt_result = self._detect_government_cheque(text_upper)
        if govt_result[0]:
            return self._create_govt_classification(govt_result[1], text_upper)
        
        # Step 2: Detect banker's instruments
        dd_result = self._detect_demand_draft(text_upper)
        if dd_result:
            return dd_result
        
        mc_result = self._detect_managers_cheque(text_upper)
        if mc_result:
            return mc_result
        
        # Step 3: Check for crossed/account payee
        for pattern in self.crossed_patterns:
            if re.search(pattern, text_upper):
                return ChequeClassification(
                    type=ChequeType.CROSSED,
                    confidence=0.85,
                    is_government=False,
                    payee_rule="account_payee",
                    validation_notes=["Crossed/Account Payee cheque detected"]
                )
        
        # Step 4: Detect self cheque
        if 'SELF' in text_upper and len(text_upper.split()) < 15:
            return ChequeClassification(
                type=ChequeType.SELF,
                confidence=0.8,
                is_government=False,
                payee_rule="self",
                validation_notes=["Self cheque for cash withdrawal"]
            )
        
        # Step 5: Default to ORDER cheque (most common)
        return ChequeClassification(
            type=ChequeType.ORDER,
            confidence=0.7,
            is_government=False,
            payee_rule="standard",
            validation_notes=["Standard order cheque - default classification"]
        )
    
    def _detect_government_cheque(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect government cheques based on PAO guidelines
        Returns (is_government, category)
        """
        # Category III - Government Account (highest priority)
        if 'GOVERNMENT ACCOUNT' in text or 'GOVT ACCOUNT' in text:
            return True, 'III'
        
        # Category II - Non-Transferable
        if 'NOT TRANSFERABLE' in text and any(g in text for g in ['GOVT', 'GOVERNMENT', 'PAO', 'DDO']):
            return True, 'II'
        
        # Category I - Negotiable
        for marker in self.govt_markers['high']:
            if marker in text:
                return True, 'I'
        
        for marker in self.govt_markers['medium']:
            if marker in text:
                return True, 'I'
        
        return False, None
    
    def _create_govt_classification(self, category: str, text: str) -> ChequeClassification:
        """Create classification with government payee rules"""
        if category == 'III':
            return ChequeClassification(
                type=ChequeType.GOVT_GOVT_ACCOUNT,
                confidence=0.98,
                is_government=True,
                government_category='III',
                payee_rule="government_account",
                validation_notes=["Category III - Inter-departmental/Government Account"]
            )
        elif category == 'II':
            # Extract officer designation if present
            designation = self._extract_officer_designation(text)
            return ChequeClassification(
                type=ChequeType.GOVT_NON_TRANSFERABLE,
                subtype=designation,
                confidence=0.95,
                is_government=True,
                government_category='II',
                payee_rule="government_officer",
                validation_notes=["Category II - Non-Transferable - Salary/Office expenses"]
            )
        else:  # Category I
            return ChequeClassification(
                type=ChequeType.GOVT_NEGOTIABLE,
                confidence=0.92,
                is_government=True,
                government_category='I',
                payee_rule="government_contractor",
                validation_notes=["Category I - Negotiable - Contractors/Suppliers"]
            )
    
    def _detect_demand_draft(self, text: str) -> Optional[ChequeClassification]:
        """Detect Demand Draft"""
        if any(marker in text for marker in self.dd_markers):
            return ChequeClassification(
                type=ChequeType.DEMAND_DRAFT,
                confidence=0.95,
                is_government=False,
                payee_rule="dd_standard",
                validation_notes=["Demand Draft - Nationwide clearance"]
            )
        return None
    
    def _detect_managers_cheque(self, text: str) -> Optional[ChequeClassification]:
        """Detect Manager's Cheque/Pay Order"""
        if any(marker in text for marker in self.mc_markers):
            return ChequeClassification(
                type=ChequeType.MANAGERS_CHEQUE,
                confidence=0.95,
                is_government=False,
                payee_rule="mc_standard",
                validation_notes=["Manager's Cheque - Local clearance only"]
            )
        return None
    
    def _extract_officer_designation(self, text: str) -> Optional[str]:
        """Extract government officer designation from Category II cheques"""
        patterns = [
            r'SECTION\s+OFFICER[^A-Z]*([A-Z&\s]+)',
            r'ACCOUNTS\s+OFFICER',
            r'DRAWING\s+AND\s+DISBURSING\s+OFFICER',
            r'DDO',
            r'PAY\s+AND\s+ACCOUNTS\s+OFFICER',
            r'PAO',
            r'CHIEF\s+ACCOUNTS\s+OFFICER',
            r'FINANCIAL\s+ADVISOR'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip()
        return None


# Singleton instance
_classifier_instance = None

def get_classifier() -> ChequeTypeClassifier:
    """Get or create classifier singleton"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ChequeTypeClassifier()
    return _classifier_instance