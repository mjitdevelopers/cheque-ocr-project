#!/usr/bin/env python3
"""
Pre-download all AI models before running the main script
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download all required AI models"""
    
    logger.info("🚀 Downloading AI Models...")
    
    # Model 1: NER Model (for entity extraction)
    logger.info("📥 Downloading NER model (dslim/distilbert-NER)...")
    ner = pipeline(
        "ner",
        model="dslim/distilbert-NER",
        device=-1  # CPU for download
    )
    logger.info("✅ NER model downloaded")
    
    # Model 2: Alternative model (backup)
    logger.info("📥 Downloading backup NER model (dbmdz/bert-large-cased-finetuned-conll03-english)...")
    ner2 = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        device=-1
    )
    logger.info("✅ Backup model downloaded")
    
    # Model 3: Zero-shot classifier
    logger.info("📥 Downloading zero-shot model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
    logger.info("✅ Zero-shot model downloaded")
    
    logger.info("\n🎉 All AI models downloaded successfully!")
    logger.info("Models are cached in: ~/.cache/huggingface/")

if __name__ == "__main__":
    download_models()