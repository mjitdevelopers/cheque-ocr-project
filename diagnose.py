# diagnose.py - Find out what's wrong

import os
import time
import psutil
from pathlib import Path

print("="*60)
print("🔬 DIAGNOSING YOUR OCR SYSTEM")
print("="*60)

# Check CPU
cores = psutil.cpu_count(logical=True)
physical = psutil.cpu_count(logical=False)
print(f"✅ CPU: {physical} physical cores, {cores} logical threads")

# Check RAM
ram = psutil.virtual_memory()
print(f"✅ RAM: {ram.total / (1024**3):.1f} GB total, {ram.available / (1024**3):.1f} GB available")

# Check if files exist
tiff_files = list(Path("cheque-images").glob("*.tiff")) + list(Path("cheque-images").glob("*.tif"))
print(f"✅ Found {len(tiff_files)} TIFF files")

# Test 1 image manually
print("\n📝 Testing 1 image manually...")
from ocr_engine import OCREngine
from tiff_processor import get_tiff_processor

engine = OCREngine(cpu_threads=2)
processor = get_tiff_processor()

if tiff_files:
    test_file = tiff_files[0]
    print(f"Testing: {test_file.name}")
    
    start = time.time()
    img, metadata = processor.preprocess(test_file)
    result = engine.process_cheque(img, str(test_file))
    elapsed = time.time() - start
    
    print(f"⏱️  Time for 1 image: {elapsed:.2f} seconds")
    print(f"⚡ Projected speed: {60/elapsed:.1f} images/minute")
    print(f"✅ Success: {result.get('success')}")
    print(f"📝 Payee: {result.get('payee_cleaned')}")
else:
    print("❌ No test images found!")

print("="*60)