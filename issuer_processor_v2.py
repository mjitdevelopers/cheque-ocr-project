import os
import csv
import argparse
from pathlib import Path
from datetime import datetime

import cv2
from paddleocr import PaddleOCR
import dbf


class IssuerBatchProcessor:
    def __init__(self, args):
        self.input_dir = Path(args.input_dir)
        self.limit = args.limit
        self.dbf_path = Path(args.dbf_path)
        self.threshold = args.threshold

        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            use_gpu=False,
            use_angle_cls=True,
            lang="en",
            show_log=False
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("issuer_output")
        output_dir.mkdir(exist_ok=True)

        self.csv_path = output_dir / f"ISSUER_RESULTS_{timestamp}.csv"

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "issuer_name", "confidence", "status"])

        print(f"CSV Created: {self.csv_path}")

        if not self.dbf_path.exists():
            raise FileNotFoundError("DBF file not found.")

        self.table = dbf.Table(str(self.dbf_path))
        self.table.open(mode=dbf.READ_WRITE)

        print("DBF Opened Successfully.")

    # --------------------------------------------------
    # ISSUER EXTRACTION
    # --------------------------------------------------
    def extract_issuer(self, ocr_result):

        if not ocr_result or not ocr_result[0]:
            return "", 0.0

        reject_phrases = [
            "please sign",
            "authorised sign",
            "authorized sign",
            "signatory",
            "signature",
            "bank",
            "account",
            "a/c",
            "rupees",
            "date"
        ]

        candidates = []

        for line in ocr_result[0]:
            text = line[1][0].strip()
            conf = float(line[1][1])
            lower_text = text.lower()

            if len(text) < 6:
                continue

            if any(r in lower_text for r in reject_phrases):
                continue

            digit_ratio = sum(c.isdigit() for c in text) / len(text)
            if digit_ratio > 0.2:
                continue

            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.6:
                continue

            uppercase_ratio = sum(c.isupper() for c in text) / len(text)
            if uppercase_ratio < 0.5:
                continue

            candidates.append((conf, text))

        if not candidates:
            return "", 0.0

        candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

        return candidates[0][1], candidates[0][0]

    # --------------------------------------------------
    # IMAGE PROCESSING
    # --------------------------------------------------
    def process_image(self, image_path):

        img = cv2.imread(str(image_path))

        if img is None:
            return "", 0.0

        h, w, _ = img.shape

        cropped = img[int(h * 0.60):h, int(w * 0.45):w]

        result = self.ocr.ocr(cropped, cls=True)

        issuer, confidence = self.extract_issuer(result)

        return issuer, confidence

    # --------------------------------------------------
    # DBF UPDATE
    # --------------------------------------------------
    def update_dbf(self, image_name, issuer):

        image_base = Path(image_name).stem.strip().lower()

        for record in self.table:
            dbf_image = str(record["IMAGE_FILE"]).strip().lower()
            dbf_base = Path(dbf_image).stem.strip().lower()

            if dbf_base == image_base:
                with record:
                    record.DRAWER_NM = issuer[:50]
                return True

        return False

    # --------------------------------------------------
    # MAIN RUN
    # --------------------------------------------------
    def run(self):

        if not self.input_dir.exists():
            print("Input directory not found.")
            return

        images = sorted([
            p for p in self.input_dir.iterdir()
            if p.suffix.lower() in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
        ])

        if self.limit:
            images = images[:self.limit]

        print(f"\nProcessing {len(images)} images...\n")

        for img_path in images:
            try:
                issuer, confidence = self.process_image(img_path)

                if confidence >= self.threshold and issuer:
                    status = "AUTO-UPDATED"
                    self.update_dbf(img_path.name, issuer)
                elif issuer:
                    status = "REVIEW"
                else:
                    status = "NO TEXT"

                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        img_path.name,
                        issuer,
                        round(confidence, 4),
                        status
                    ])

                print(f"{img_path.name} -> {issuer} ({round(confidence,2)}) [{status}]")

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        self.table.close()
        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--dbf-path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.85)

    args = parser.parse_args()

    processor = IssuerBatchProcessor(args)
    processor.run()