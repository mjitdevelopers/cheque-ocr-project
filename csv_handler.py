"""
CSV/TXT Handler for Cheque Processing Results
Creates Notepad-friendly output files
Last Updated: 2025
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CSVHandler:
    """
    Handle CSV/TXT file operations for cheque processing results
    All outputs are plain text - open with Notepad or Excel
    """
    
    def __init__(self):
        self.field_names = [
            'IMAGE_ID',          # Unique identifier
            'FILENAME',          # Original filename
            'CHEQUE_TYPE',       # Bearer/Order/DD/Govt/etc.
            'IS_GOVERNMENT',     # Yes/No
            'GOVT_CATEGORY',     # I/II/III
            'PAYEE_RAW',         # Raw extracted text
            'PAYEE_CLEANED',     # Final cleaned payee name
            'CONFIDENCE',        # OCR confidence score
            'PROCESS_DATE',      # Timestamp
            'PROCESS_TIME_MS',   # Processing time
            'STATUS',            # SUCCESS/FAILED
            'ERROR_MESSAGE',     # Error details if failed
        ]
    
    def create_csv(self, output_path: str) -> bool:
        """
        Create a new CSV file with headers
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.field_names)
            logger.info(f"Created CSV file: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create CSV: {e}")
            return False
    
    def write_record(self, csv_path: str, record: Dict[str, Any]) -> bool:
        """
        Append a single record to CSV file
        """
        try:
            # Convert record to list in correct order
            row = []
            for field in self.field_names:
                value = record.get(field, '')
                # Handle different data types
                if isinstance(value, (int, float)):
                    row.append(str(value))
                elif value is None:
                    row.append('')
                else:
                    row.append(str(value))
            
            # Append to CSV
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write CSV record: {e}")
            return False
    
    def write_batch(self, csv_path: str, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Write multiple records to CSV
        """
        stats = {'total': len(records), 'success': 0, 'failed': 0}
        
        try:
            # Write all records at once for efficiency
            rows = []
            for record in records:
                row = []
                for field in self.field_names:
                    value = record.get(field, '')
                    if isinstance(value, (int, float)):
                        row.append(str(value))
                    elif value is None:
                        row.append('')
                    else:
                        row.append(str(value))
                rows.append(row)
                stats['success'] += 1
            
            # Append all rows
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
        except Exception as e:
            logger.error(f"Failed to write batch: {e}")
            stats['failed'] = len(records) - stats['success']
        
        return stats
    
    def create_summary_txt(self, output_dir: Path, stats: Dict[str, Any]) -> str:
        """
        Create a human-readable summary TXT file
        Perfect for opening in Notepad
        """
        summary_path = output_dir / "results_summary.txt"
        
        lines = [
            "=" * 50,
            "CHEQUE OCR PROCESSING SUMMARY",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 STATISTICS",
            "-" * 30,
            f"Total images processed: {stats.get('total', 0)}",
            f"Successful: {stats.get('successful', 0)}",
            f"Failed: {stats.get('failed', 0)}",
            f"Government cheques: {stats.get('government', 0)}",
            "",
            "⚡ PERFORMANCE",
            "-" * 30,
            f"Total time: {stats.get('total_time_min', 0)} minutes",
            f"Throughput: {stats.get('throughput_per_min', 0)} images/minute",
            f"Avg time per image: {stats.get('avg_time_ms', 0):.1f} ms",
            "",
            "📁 OUTPUT FILES",
            "-" * 30,
            f"Main results: results.csv",
            f"This summary: results_summary.txt",
            f"Failed images: failed_images.txt",
            "",
            "=" * 50,
            "END OF SUMMARY",
            "=" * 50
        ]
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return str(summary_path)
        except Exception as e:
            logger.error(f"Failed to write summary: {e}")
            return ""
    
    def create_failed_list(self, output_dir: Path, failed_records: List[Dict]) -> str:
        """
        Create a simple TXT file listing failed images
        """
        failed_path = output_dir / "failed_images.txt"
        
        lines = ["FAILED IMAGES LIST", "=" * 30, ""]
        
        for record in failed_records:
            lines.append(f"File: {record.get('FILENAME', 'Unknown')}")
            lines.append(f"Error: {record.get('ERROR_MESSAGE', 'Unknown')}")
            lines.append("-" * 20)
        
        try:
            with open(failed_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return str(failed_path)
        except Exception as e:
            logger.error(f"Failed to write failed list: {e}")
            return ""
    
    def read_csv(self, csv_path: str) -> List[Dict]:
        """
        Read records from CSV file
        """
        records = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
            return records
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return []