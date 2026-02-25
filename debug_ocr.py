from paddleocr import PaddleOCR
from pathlib import Path
import cv2

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)

# Folder with images
folder = "F_23022026_010"

# Pehle 5 images check karo
images = list(Path(folder).glob("*.tiff"))[:5]

for img_path in images:
    print("\n" + "="*80)
    print(f"📄 Image: {img_path.name}")
    print("="*80)
    
    # Run OCR
    result = ocr.ocr(str(img_path), cls=True)
    
    if result and result[0]:
        print("\n📝 OCR TEXT FOUND:")
        for i, line in enumerate(result[0]):
            text = line[1][0]
            confidence = line[1][1]
            print(f"{i+1:2d}. {text:60} (conf: {confidence:.2f})")
    else:
        print("❌ NO TEXT FOUND!")