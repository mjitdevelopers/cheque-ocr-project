import os
import csv
import argparse
from pathlib import Path
from datetime import datetime

from paddleocr import PaddleOCR


class IssuerBatchProcessor:
    def __init__(self, args):
        self.input_dir = Path(args.input_dir)
        self.limit = args.limit

        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            use_gpu=False,        # CPU mode (stable)
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
            writer.writerow(["image_name", "issuer_name", "confidence"])

        print(f"CSV Created: {self.csv_path}")

    def extract_issuer(self, ocr_result):
        """
        Simple heuristic:
        Take highest confidence text line with max length.
        You can improve this later.
        """
        if not ocr_result or not ocr_result[0]:
            return "", 0.0

        best_text = ""
        best_conf = 0.0

        for line in ocr_result[0]:
            text = line[1][0].strip()
            conf = float(line[1][1])

            if len(text) > 3 and conf > best_conf:
                best_text = text
                best_conf = conf

        return best_text, best_conf

    def process_image(self, image_path):
        result = self.ocr.ocr(str(image_path), cls=True)
        issuer, confidence = self.extract_issuer(result)
        return issuer, confidence

    def run(self):
        if not self.input_dir.exists():
            print("Input directory not found.")
            return

        images = sorted([
            p for p in self.input_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        # 🔥 LIMIT APPLIED HERE
        if self.limit:
            images = images[:self.limit]

        print(f"\nProcessing {len(images)} images...\n")

        for img_path in images:
            try:
                issuer, confidence = self.process_image(img_path)

                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        img_path.name,
                        issuer,
                        round(confidence, 4)
                    ])

                print(f"✔ {img_path.name} -> {issuer} ({round(confidence, 2)})")

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Input image folder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")

    args = parser.parse_args()

    processor = IssuerBatchProcessor(args)
    processor.run()