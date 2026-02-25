import os
from pathlib import Path
import csv
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)

# Input folder
input_folder = "F_23022026_010"
output_file = "issuer_names.csv"

# Get all images
images = []
for ext in ['*.tif', '*.tiff', '*.jpg', '*.png']:
    images.extend(Path(input_folder).glob(ext))

print(f"Found {len(images)} images")

# Process and save
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Filename', 'Issuer_Name'])
    
    for img in images:
        print(f"Processing: {img.name}")
        
        # OCR
        result = ocr.ocr(str(img), cls=True)
        
        # Get text
        text = ""
        if result and result[0]:
            for line in result[0]:
                text += line[1][0] + " "
        
        # Simple extraction - look for "FOR"
        issuer = "UNKNOWN"
        if "FOR" in text.upper():
            parts = text.upper().split("FOR")
            if len(parts) > 1:
                issuer = parts[1].strip().split()[0:3]
                issuer = " ".join(issuer)
        
        writer.writerow([img.name, issuer])
        print(f"  Issuer: {issuer}")

print(f"\nDone! Results saved to {output_file}")