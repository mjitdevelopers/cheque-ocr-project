# test_imports.py
import sys
print(f"Python: {sys.version}")
print(f"Frozen: {getattr(sys, 'frozen', False)}")

print("\nTrying to import paddleocr...")
try:
    from paddleocr import PaddleOCR
    print("✅ paddleocr imported successfully")
except Exception as e:
    print(f"❌ Failed: {e}")

input("\nPress Enter to exit...")