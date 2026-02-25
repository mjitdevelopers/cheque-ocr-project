#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Government Payee Rule Engine
Enforces strict payee naming rules for government cheques
Based on PAO Accounting Standards Chapter 2 & CAM-11 Guidelines
Last Updated: 2025
"""

import re
from typing import Optional, List, Dict
from cheque_classifier import ChequeClassification, ChequeType
import logging

logger = logging.getLogger(__name__)


class GovernmentPayeeRuleEngine:
    """
    Enforces government cheque payee rules:
    - Category III (Government Account): ALWAYS "GOVERNMENT"
    - Category II (Non-Transferable): Payee is GOVERNMENT OFFICER BY DESIGNATION
    - Category I (Negotiable): Contractors/Suppliers, but if to govt -> "GOVERNMENT"
    - DD/MC issued TO government: Payee = "GOVERNMENT"
    """
    
    def __init__(self):
        # Government entity identifiers (comprehensive)
        self.govt_entities = [
            # Central Government
            'GOVERNMENT OF INDIA', 'GOVT OF INDIA', 'CENTRAL GOVERNMENT',
            'INCOME TAX DEPARTMENT', 'CUSTOMS DEPARTMENT', 'GST DEPARTMENT',
            'CENTRAL EXCISE', 'RAILWAYS', 'INDIAN RAILWAYS',
            'DEFENCE', 'INDIAN ARMY', 'INDIAN NAVY', 'INDIAN AIR FORCE',
            'POSTAL DEPARTMENT', 'INDIA POST', 'TELEGRAPH DEPARTMENT',
            'PUBLIC WORKS DEPARTMENT', 'PWD', 'CPWD', 'NPWD',
            'CENTRAL PUBLIC WORKS', 'CENTRAL WATER COMMISSION',
            
            # State Government
            'STATE GOVERNMENT', 'GOVT OF', 'SARKAR',
            'ELECTRICITY BOARD', 'ELECTRICITY DEPARTMENT', 'POWER CORPORATION',
            'WATER BOARD', 'JAL BOARD', 'IRRIGATION DEPARTMENT',
            'PUBLIC HEALTH ENGINEERING', 'PHE',
            'NAGAR PALIKA', 'MUNICIPAL CORPORATION', 'MUNICIPAL COUNCIL',
            'ZILLA PARISHAD', 'PANCHAYAT', 'GRAM PANCHAYAT',
            'FOREST DEPARTMENT', 'FISHERIES DEPARTMENT',
            'AGRICULTURE DEPARTMENT', 'HORTICULTURE DEPARTMENT',
            'HEALTH DEPARTMENT', 'MEDICAL SERVICES', 'CIVIL HOSPITAL',
            'EDUCATION DEPARTMENT', 'SCHOOL EDUCATION', 'HIGHER EDUCATION',
            'SOCIAL WELFARE', 'WOMEN AND CHILD DEVELOPMENT',
            'TRANSPORT DEPARTMENT', 'ROAD TRANSPORT', 'RTO',
            'PROPERTY TAX', 'HOUSE TAX', 'LAND REVENUE',
            
            # Government Bodies
            'MUNICIPALITY', 'MUNICIPAL BOARD', 'CANTONMENT BOARD',
            'DEVELOPMENT AUTHORITY', 'HOUSING BOARD', 'SLUM BOARD',
            'IMPROVEMENT TRUST', 'CITY TRUST',
            'PORT TRUST', 'PORT AUTHORITY', 'AIRPORT AUTHORITY',
            'TOURISM DEVELOPMENT', 'INDUSTRIAL DEVELOPMENT',
            
            # PSUs (if treated as government for payee purposes)
            'BANK OF INDIA', 'STATE BANK OF INDIA', 'SBI', 'CANARA BANK',
            'PUNJAB NATIONAL BANK', 'PNB', 'BANK OF BARODA', 'BOB',
            'UNION BANK', 'INDIAN BANK', 'CENTRAL BANK',
            'LIFE INSURANCE CORPORATION', 'LIC', 'GIC',
            'OIL AND NATURAL GAS', 'ONGC', 'INDIAN OIL', 'IOC',
            'BHARAT PETROLEUM', 'BPCL', 'HINDUSTAN PETROLEUM', 'HPCL',
            'GAIL', 'POWER GRID', 'NTPC', 'NHPC', 'SJVN',
            'BHEL', 'HAL', 'BEL', 'BEML', 'COAL INDIA', 'SAIL',
            'NMDC', 'HCL', 'NALCO', 'MOIL'
        ]
        
        # Patterns that indicate payee is government
        self.govt_payee_patterns = [
            r'PAY[:\s]*GOVT',
            r'PAY[:\s]*GOVERNMENT',
            r'FAVOUR[:\s]*GOVT',
            r'FAVOUR[:\s]*GOVERNMENT',
            r'TO\s+THE\s+([A-Z\s]+DEPARTMENT)',
            r'([A-Z\s]+BOARD)',
            r'([A-Z\s]+MUNICIPALITY)',
            r'([A-Z\s]+COMMITTEE)',
            r'([A-Z\s]+AUTHORITY)',
            r'([A-Z\s]+TRUST)',
            r'([A-Z\s]+CORPORATION)',
        ]
        
        # Officer designations for Category II
        self.officer_designations = [
            'SECTION OFFICER',
            'ACCOUNTS OFFICER',
            'FINANCE OFFICER',
            'DRAWING AND DISBURSING OFFICER',
            'DDO',
            'PAY AND ACCOUNTS OFFICER',
            'PAO',
            'CHIEF ACCOUNTS OFFICER',
            'FINANCIAL ADVISOR',
            'CONTROLLER OF ACCOUNTS',
            'DEPUTY CONTROLLER',
            'ASSISTANT CONTROLLER',
            'SENIOR ACCOUNTS OFFICER',
            'JUNIOR ACCOUNTS OFFICER',
            'ACCOUNTS ASSISTANT',
            'TREASURY OFFICER',
            'DEPUTY TREASURY OFFICER',
            'ASSISTANT TREASURY OFFICER',
            'BANK OFFICER',
            'SENIOR MANAGER',
            'CHIEF MANAGER',
            'GENERAL MANAGER',
            'DEPUTY GENERAL MANAGER',
            'ASSISTANT GENERAL MANAGER',
            'EXECUTIVE DIRECTOR',
            'DIRECTOR',
            'JOINT DIRECTOR',
            'DEPUTY DIRECTOR',
            'ASSISTANT DIRECTOR',
            'UNDER SECRETARY',
            'DEPUTY SECRETARY',
            'JOINT SECRETARY',
            'ADDITIONAL SECRETARY',
            'PRINCIPAL SECRETARY',
            'CHIEF SECRETARY'
        ]
    
    def enforce_payee_rule(self,
                           classification: ChequeClassification,
                           extracted_payee: Optional[str],
                           ocr_full_text: str) -> str:
        """
        ENFORCE GOVERNMENT PAYEE RULES.
        Returns the CORRECT payee name according to regulations.
        """
        if not extracted_payee:
            extracted_payee = ""
        
        # Rule 1: Category III - Government Account (Highest priority)
        if classification.type == ChequeType.GOVT_GOVT_ACCOUNT:
            logger.debug("Category III: Setting payee to GOVERNMENT")
            return "GOVERNMENT"
        
        # Rule 2: Category II - Non-Transferable
        if classification.type == ChequeType.GOVT_NON_TRANSFERABLE:
            # Extract officer designation
            designation = self._extract_officer_designation(ocr_full_text)
            if designation:
                return f"GOVERNMENT - {designation}"
            # Check if extracted payee contains designation
            if extracted_payee and any(d in extracted_payee.upper() for d in self.officer_designations):
                return f"GOVERNMENT - {extracted_payee}"
            return "GOVERNMENT OFFICER"
        
        # Rule 3: Category I - Negotiable but check if payee is government
        if classification.type == ChequeType.GOVT_NEGOTIABLE:
            if self._is_government_entity(extracted_payee):
                return "GOVERNMENT"
            return extracted_payee or "XXX"
        
        # Rule 4: DD/MC issued TO government
        if classification.type in [ChequeType.DEMAND_DRAFT, ChequeType.MANAGERS_CHEQUE]:
            if self._is_payable_to_government(ocr_full_text):
                return "GOVERNMENT"
            return extracted_payee or "XXX"
        
        # Rule 5: Crossed/Account Payee cheque to government
        if self._is_payable_to_government(ocr_full_text):
            return "GOVERNMENT"
        
        # Rule 6: Any cheque with government in payee name
        if self._is_government_entity(extracted_payee):
            return "GOVERNMENT"
        
        # Default: return extracted payee
        return extracted_payee or "XXX"
    
    def _is_government_entity(self, text: str) -> bool:
        """Check if text indicates a government entity"""
        if not text:
            return False
        
        text_upper = text.upper()
        
        # Direct match with government entities
        for entity in self.govt_entities:
            if entity in text_upper:
                return True
        
        # Pattern-based detection
        for pattern in self.govt_payee_patterns:
            if re.search(pattern, text_upper):
                return True
        
        # Check for common government name patterns
        govt_indicators = ['GOVT', 'GOVERNMENT', 'MUNICIPAL', 'NAGAR', 'GRAM',
                          'ZILLA', 'PANCHAYAT', 'ELECTRICITY', 'WATER',
                          'IRRIGATION', 'FOREST', 'HEALTH', 'EDUCATION',
                          'TRANSPORT', 'DEVELOPMENT', 'AUTHORITY', 'BOARD',
                          'TRUST', 'CORPORATION', 'COMMITTEE']
        
        words = text_upper.split()
        if len(words) <= 4:  # Short name likely
            for indicator in govt_indicators:
                if indicator in text_upper:
                    return True
        
        return False
    
    def _is_payable_to_government(self, ocr_text: str) -> bool:
        """Detect if cheque is payable to government entity"""
        if not ocr_text:
            return False
        
        text_upper = ocr_text.upper()
        
        # Look for payee section containing government terms
        payee_section = self._extract_payee_section(text_upper)
        if payee_section:
            return self._is_government_entity(payee_section)
        
        return False
    
    def _extract_payee_section(self, text: str) -> Optional[str]:
        """Extract the portion of text that likely contains payee name"""
        # Look for common payee indicators
        indicators = ['PAY', 'PAY TO', 'ORDER OF', 'FAVOUR OF', 'BENEFICIARY']
        
        for indicator in indicators:
            if indicator in text:
                parts = text.split(indicator, 1)
                if len(parts) > 1:
                    # Take next 200 chars max
                    return parts[1][:200]
        
        return None
    
    def _extract_officer_designation(self, text: str) -> Optional[str]:
        """Extract government officer designation from text"""
        text_upper = text.upper()
        
        for designation in self.officer_designations:
            if designation in text_upper:
                return designation
        
        return None


# Singleton instance
_rule_engine_instance = None

def get_rule_engine() -> GovernmentPayeeRuleEngine:
    """Get or create rule engine singleton"""
    global _rule_engine_instance
    if _rule_engine_instance is None:
        _rule_engine_instance = GovernmentPayeeRuleEngine()
    return _rule_engine_instance
