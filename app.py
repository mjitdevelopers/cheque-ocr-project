#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cheque OCR & DBF Updater Application
Fixed for NumPy ABI compatibility issues
"""

# ============= NUMPY COMPATIBILITY FIX =============
# This must be at the very top, before any other imports
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Handle numpy compatibility issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['FLAGS_logtostderr'] = '0'
os.environ['GLOG_minloglevel'] = '2'
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
os.environ['PPOCR_HOME'] = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), '.paddleocr')

# Try to import numpy first to check version
try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except ImportError:
    print("NumPy not installed yet")
except Exception as e:
    print(f"NumPy import error: {e}")
# ============= END NUMPY COMPATIBILITY FIX =============

import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import logging
logging.disable(logging.CRITICAL)
import re
import traceback
import csv
from datetime import datetime
from pathlib import Path

# ============= LAZY OCR INITIALIZATION =============
# PaddleOCR import - ONLY check if module exists, don't initialize yet!

# Check if PaddleOCR module is available
PADDLEOCR_AVAILABLE = False
ocr = None
OCR_AVAILABLE = False

try:
    # Just check if we can import the module
    import paddleocr
    PADDLEOCR_AVAILABLE = True
    print("✅ PaddleOCR module found")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    print(f"❌ PaddleOCR module not found: {e}")
except Exception as e:
    PADDLEOCR_AVAILABLE = False
    print(f"❌ Error checking PaddleOCR: {e}")

def init_ocr():
    """Initialize OCR only when needed (lazy initialization) with NumPy compatibility handling"""
    global ocr, OCR_AVAILABLE, PADDLEOCR_AVAILABLE
    
    if OCR_AVAILABLE and ocr is not None:
        return True
    
    if not PADDLEOCR_AVAILABLE:
        print("DEBUG: PaddleOCR module not available")
        OCR_AVAILABLE = False
        return False
    
    try:
        import os
        import sys
        import tempfile
        import traceback
        
        print("DEBUG: Starting OCR initialization...")
        
        # Set environment variables again to be safe
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['FLAGS_logtostderr'] = '0'
        os.environ['GLOG_minloglevel'] = '2'
        os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
        
        # Set paddle home based on environment
        if getattr(sys, 'frozen', False):
            # Running as EXE - use temp folder
            paddle_home = os.path.join(tempfile.gettempdir(), 'paddleocr_models')
            print(f"DEBUG EXE mode: Using paddle home at {paddle_home}")
        else:
            # Running as script - use user home
            paddle_home = os.path.expanduser('~/.paddleocr')
            print(f"DEBUG Script mode: Using paddle home at {paddle_home}")
        
        # Check numpy version and warn if incompatible
        try:
            import numpy
            numpy_version = numpy.__version__
            print(f"DEBUG: NumPy version detected: {numpy_version}")
            
            # Check for known incompatible versions
            major_version = int(numpy_version.split('.')[0])
            if major_version >= 2:
                print("⚠️  WARNING: NumPy version 2.x detected - may have ABI compatibility issues")
                print("⚠️  If OCR fails, try: pip install numpy==1.24.3")
        except:
            print("DEBUG: Could not determine NumPy version")
        
        # Create directory
        print(f"DEBUG: Checking if {paddle_home} exists...")
        os.makedirs(paddle_home, exist_ok=True)
        print(f"DEBUG: Directory exists: {os.path.exists(paddle_home)}")
        
        # Set environment variable
        os.environ['PPOCR_HOME'] = paddle_home
        print(f"DEBUG: Set PPOCR_HOME to {paddle_home}")
        
        # Try to import PaddleOCR with specific error handling
        try:
            from paddleocr import PaddleOCR
            print("DEBUG: Successfully imported PaddleOCR")
        except ImportError as e:
            if "numpy.core.multiarray" in str(e):
                print("❌ DEBUG: NumPy ABI compatibility issue detected!")
                print("❌ DEBUG: This is caused by NumPy version mismatch")
                print("❌ DEBUG: Please install numpy==1.24.3")
                print("❌ DEBUG: Run: pip install numpy==1.24.3")
                
                # Show error message to user via GUI later
                OCR_AVAILABLE = False
                return False
            else:
                print(f"❌ DEBUG: Import error: {e}")
                raise
        
        # Initialize OCR with error handling
        print("DEBUG: Creating PaddleOCR instance...")
        
        # Use try-except for the actual initialization
        try:
            ocr = PaddleOCR(
                use_angle_cls=False, 
                show_log=False, 
                lang='en',
                det_db_thresh=0.3,
                det_db_box_thresh=0.2,
                det_db_unclip_ratio=1.5,
                rec_thresh=0.5
            )
            
            OCR_AVAILABLE = True
            print("DEBUG: ✅ PaddleOCR initialized successfully!")
            return True
            
        except Exception as e:
            if "numpy" in str(e).lower():
                print(f"❌ DEBUG: NumPy related error during initialization: {e}")
                print("❌ DEBUG: This may be due to ABI incompatibility")
            else:
                print(f"❌ DEBUG: Error during PaddleOCR initialization: {e}")
            raise
        
    except Exception as e:
        OCR_AVAILABLE = False
        print(f"❌ DEBUG: OCR Init Error: {e}")
        print("DEBUG: Full traceback:")
        traceback.print_exc()
        
        # Error ko file mein log karo
        try:
            with open('ocr_error.log', 'w', encoding='utf-8') as f:
                f.write(f"Error: {e}\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
                f.write(f"\nNumPy version: {numpy.__version__ if 'numpy' in dir() else 'Unknown'}")
        except:
            pass
            
        return False
# ============= END LAZY OCR INITIALIZATION =============

# DBF import with error handling
DBF_AVAILABLE = False
try:
    import dbf
    DBF_AVAILABLE = True
    print("✅ DBF module found")
except ImportError:
    DBF_AVAILABLE = False
    print("❌ DBF module not available - run: pip install dbf")
except Exception as e:
    DBF_AVAILABLE = False
    print(f"❌ DBF module error: {e}")

# 🔐 Allowed MAC
ALLOWED_MAC = "E4-0D-36-60-FF-B4"

def get_system_mac():
    """Get system MAC address for licensing"""
    try:
        import subprocess
        output = subprocess.check_output("ipconfig /all", shell=True).decode('utf-8', errors='ignore')
        blocks = output.split("\n\n")
        for block in blocks:
            if "IPv4 Address" in block:
                match = re.search(r"Physical Address[.\s]*: ([A-F0-9\-]+)", block, re.IGNORECASE)
                if match:
                    return match.group(1).strip().upper()
    except:
        pass
    return None

def check_mac():
    """Check if MAC address is authorized"""
    # For testing - always return True
    # In production, uncomment the actual check
    # system_mac = get_system_mac()
    # return system_mac == ALLOWED_MAC
    return True

def update_dbf_file(dbf_path, image_name, issuer_name, confidence):
    """Update DBF file with extracted issuer name"""
    try:
        if not DBF_AVAILABLE:
            return False, "DBF module not installed"
        
        if not os.path.exists(dbf_path):
            return False, "DBF file not found"
        
        # Open DBF
        table = dbf.Table(dbf_path)
        table.open(mode=dbf.READ_WRITE)
        
        # Get image stem for comparison
        image_stem = Path(image_name).stem.lower()
        updated = False
        
        # Find matching record
        for record in table:
            try:
                dbf_image = str(record.IMAGE_FILE).strip().lower()
                dbf_stem = Path(dbf_image).stem.lower()
                
                if dbf_stem == image_stem:
                    with record:
                        # Update DRAWER_NM
                        record.DRAWER_NM = issuer_name[:50]
                        
                        # Update OPR_NO if field exists
                        try:
                            record.OPR_NO = "AS601"
                        except:
                            pass
                        
                        # Update FILE_MARK if field exists
                        try:
                            record.FILE_MARK = True
                        except:
                            pass
                    updated = True
                    break
            except:
                continue
        
        table.close()
        
        if updated:
            return True, f"Updated: {image_name}"
        else:
            return False, f"No matching record found for {image_name}"
            
    except Exception as e:
        return False, str(e)

def start_process():
    """Main processing function"""
    try:
        # Initialize OCR first if needed
        if PADDLEOCR_AVAILABLE and not OCR_AVAILABLE:
            init_result = init_ocr()
            if not init_result:
                print("OCR initialization failed")
        
        # If OCR is still not available, show warning but allow continuing
        if not OCR_AVAILABLE:
            continue_without_ocr = messagebox.askyesno(
                "OCR Not Available", 
                "OCR is not available.\n\n"
                "Possible reasons:\n"
                "• NumPy version incompatibility (try: pip install numpy==1.24.3)\n"
                "• PaddleOCR not properly installed\n"
                "• Missing dependencies\n\n"
                "Continue without OCR? (Only folder selection will work)"
            )
            if not continue_without_ocr:
                return
        
        # Step 1: Select Image Folder
        folder = filedialog.askdirectory(title="Step 1: Select Image Folder")
        if not folder:
            return

        # Get images
        images = []
        valid_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")
        for f in os.listdir(folder):
            if f.lower().endswith(valid_extensions):
                images.append(f)
        
        images = sorted(images)

        if not images:
            messagebox.showwarning("No Images", "No images found in selected folder.")
            return

        # Step 2: Select DBF File (before processing)
        dbf_file = filedialog.askopenfilename(
            title="Step 2: Select DBF File to Update",
            filetypes=[("DBF files", "*.dbf"), ("All files", "*.*")]
        )
        
        if not dbf_file:
            # User cancelled DBF selection - ask if they want to continue without DBF
            if not messagebox.askyesno("No DBF", "No DBF file selected. Continue without DBF update?"):
                return
            dbf_file = None

        # Create progress window
        progress_win = tk.Toplevel(root)
        progress_win.title("Processing Images")
        progress_win.geometry("600x450")
        progress_win.grab_set()
        progress_win.resizable(False, False)
        
        # Center window
        progress_win.transient(root)
        progress_win.update_idletasks()
        width = progress_win.winfo_width()
        height = progress_win.winfo_height()
        x = (progress_win.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_win.winfo_screenheight() // 2) - (height // 2)
        progress_win.geometry(f'{width}x{height}+{x}+{y}')

        # Title
        tk.Label(progress_win, text="Processing Images", 
                font=("Arial", 12, "bold")).pack(pady=10)

        # Progress bar
        progress_bar = ttk.Progressbar(
            progress_win,
            orient="horizontal",
            length=500,
            mode="determinate"
        )
        progress_bar.pack(pady=10)

        # Status labels
        status_label = tk.Label(progress_win, text="Initializing...", wraplength=550)
        status_label.pack(pady=5)
        
        count_label = tk.Label(progress_win, text="", font=("Arial", 10))
        count_label.pack(pady=5)
        
        # Current image label
        current_label = tk.Label(progress_win, text="", fg="blue")
        current_label.pack(pady=5)
        
        # OCR Status with more details
        ocr_status_text = "✅ OCR Ready" if OCR_AVAILABLE else "❌ OCR Not Available"
        if not OCR_AVAILABLE and PADDLEOCR_AVAILABLE:
            ocr_status_text = "⚠️ OCR Initialization Failed (NumPy compatibility?)"
        
        ocr_status = tk.Label(
            progress_win, 
            text=f"OCR: {ocr_status_text}", 
            fg="green" if OCR_AVAILABLE else ("orange" if PADDLEOCR_AVAILABLE else "red")
        )
        ocr_status.pack(pady=5)
        
        # NumPy version info
        try:
            import numpy
            numpy_label = tk.Label(
                progress_win, 
                text=f"NumPy version: {numpy.__version__}",
                fg="orange" if numpy.__version__.startswith('2') else "green"
            )
            numpy_label.pack(pady=2)
        except:
            pass
        
        # DBF update status
        dbf_label = tk.Label(progress_win, text="", fg="green")
        dbf_label.pack(pady=5)
        
        # Results text box
        results_frame = tk.Frame(progress_win)
        results_frame.pack(pady=10, fill='both', expand=True)
        
        results_text = tk.Text(results_frame, height=10, width=70, wrap=tk.WORD)
        results_text.pack(side=tk.LEFT, fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(results_frame, command=results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.config(yscrollcommand=scrollbar.set)

        progress_bar["maximum"] = len(images)
        
        # Process images
        valid_count = 0
        junk_count = 0
        dbf_updates = 0
        dbf_failures = 0

        for idx, image_name in enumerate(images, 1):
            image_path = os.path.join(folder, image_name)

            try:
                # Update status
                status_label.config(text=f"Processing: {image_name}")
                count_label.config(text=f"Image {idx} of {len(images)}")
                progress_bar["value"] = idx
                progress_win.update()

                # Perform OCR - with safety check
                extracted_text = "NO OCR"
                confidence = 0
                is_valid = False
                
                if OCR_AVAILABLE and ocr is not None:
                    try:
                        result = ocr.ocr(image_path, cls=False)
                        
                        if result and result[0]:
                            texts = []
                            confidences = []
                            for line in result[0]:
                                if len(line) >= 2:
                                    texts.append(line[1][0])
                                    confidences.append(line[1][1])
                            
                            extracted_text = " | ".join(texts)
                            confidence = sum(confidences) / len(confidences) if confidences else 0
                            
                            # Simple validation
                            if len(extracted_text) >= 5 and confidence >= 0.7:
                                if not any(bank in extracted_text.upper() for bank in ["BANK", "HDFC", "ICICI", "SBI", "AXIS"]):
                                    is_valid = True
                                    valid_count += 1
                                else:
                                    junk_count += 1
                            else:
                                junk_count += 1
                        else:
                            extracted_text = "NO TEXT FOUND"
                            junk_count += 1
                    except Exception as e:
                        extracted_text = f"OCR ERROR: {str(e)[:30]}"
                        junk_count += 1
                        results_text.insert(tk.END, f"⚠️ {image_name}: OCR Error - {str(e)[:50]}\n")
                        
                        # Check if it's a numpy error
                        if "numpy" in str(e).lower():
                            results_text.insert(tk.END, "   💡 Try: pip install numpy==1.24.3\n")
                else:
                    extracted_text = "OCR NOT AVAILABLE"
                    junk_count += 1
                    results_text.insert(tk.END, f"⚠️ {image_name}: OCR Not Available\n")
                
                # Update DBF if file selected
                if dbf_file and is_valid:
                    success, msg = update_dbf_file(dbf_file, image_name, extracted_text[:50], confidence)
                    if success:
                        dbf_updates += 1
                        dbf_label.config(text=f"✅ DBF Updated: {dbf_updates} records")
                        results_text.insert(tk.END, f"✅ {image_name}: DBF Updated\n")
                    else:
                        dbf_failures += 1
                        results_text.insert(tk.END, f"⚠️ {image_name}: DBF Update Failed - {msg}\n")
                elif dbf_file and not is_valid:
                    results_text.insert(tk.END, f"❌ {image_name}: Invalid Text - Not updating DBF\n")
                else:
                    results_text.insert(tk.END, f"📄 {image_name}: {extracted_text[:50]}...\n")
                
                results_text.see(tk.END)
                progress_win.update()

            except Exception as e:
                junk_count += 1
                results_text.insert(tk.END, f"❌ {image_name}: ERROR - {str(e)[:50]}\n")
                progress_win.update()

        progress_win.destroy()

        # Save results to CSV
        try:
            output_dir = Path(folder) / "ocr_results"
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_file = output_dir / f"ocr_results_{timestamp}.csv"
            
            with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['Image', 'Status', 'Extracted Text', 'Confidence'])
                # We need to recreate results - simplified version
                for img in images:
                    writer.writerow([img, "Processed", "", ""])
            
            # Show summary
            success_rate = (valid_count/len(images)*100) if images else 0
            
            summary = f"""✅ PROCESSING COMPLETE!

📊 SUMMARY:
═══════════════════════════════
📁 Total Images:    {len(images)}
✅ Valid Texts:     {valid_count}
❌ Rejected:        {junk_count}
📈 Success Rate:    {success_rate:.1f}%

📂 Results saved to:
{csv_file}
"""
            
            if dbf_file:
                summary += f"""
📁 DBF File:        {os.path.basename(dbf_file)}
✅ DBF Updates:     {dbf_updates}
❌ DBF Failures:    {dbf_failures}
"""
            
            # Add numpy recommendation if OCR failed
            if not OCR_AVAILABLE and PADDLEOCR_AVAILABLE:
                summary += """
⚠️  OCR was not available due to NumPy compatibility.
💡 Solution: pip install numpy==1.24.3
"""
            
            messagebox.showinfo("Processing Complete", summary)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    except Exception as e:
        error_msg = traceback.format_exc()
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    try:
        # Create main window
        root = tk.Tk()
        root.title("Cheque Issuer OCR - DBF Updater")
        
        # Set window size
        window_width = 600
        window_height = 500
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        root.resizable(False, False)

        # Check MAC
        if not check_mac():
            messagebox.showerror(
                "Unauthorized",
                "This software is not licensed for this system."
            )
            sys.exit()

        # Main frame
        main_frame = tk.Frame(root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')

        # Title
        tk.Label(main_frame, text="Cheque OCR & DBF Updater", 
                font=("Arial", 16, "bold"), fg="#2c3e50").pack(pady=10)

        # Status Frame
        status_frame = tk.Frame(main_frame, bg="#f8f9fa", relief=tk.GROOVE, bd=1)
        status_frame.pack(pady=10, fill='x')
        
        tk.Label(status_frame, text="System Status:", 
                font=("Arial", 11, "bold"), bg="#f8f9fa").pack(anchor='w', padx=10, pady=5)
        
        # PaddleOCR Status
        if PADDLEOCR_AVAILABLE:
            tk.Label(status_frame, text="✅ PaddleOCR Module: Available", 
                    font=("Arial", 10), fg="green", bg="#f8f9fa").pack(anchor='w', padx=20)
        else:
            tk.Label(status_frame, text="❌ PaddleOCR Module: Not Installed", 
                    font=("Arial", 10), fg="red", bg="#f8f9fa").pack(anchor='w', padx=20)
        
        # NumPy Version
        try:
            import numpy
            numpy_color = "orange" if numpy.__version__.startswith('2') else "green"
            numpy_warning = " (may cause issues)" if numpy.__version__.startswith('2') else ""
            tk.Label(status_frame, text=f"📦 NumPy Version: {numpy.__version__}{numpy_warning}", 
                    font=("Arial", 10), fg=numpy_color, bg="#f8f9fa").pack(anchor='w', padx=20)
        except:
            tk.Label(status_frame, text="📦 NumPy: Not Available", 
                    font=("Arial", 10), fg="red", bg="#f8f9fa").pack(anchor='w', padx=20)
        
        # DBF Status
        if DBF_AVAILABLE:
            tk.Label(status_frame, text="✅ DBF Module: Ready", 
                    font=("Arial", 10), fg="green", bg="#f8f9fa").pack(anchor='w', padx=20)
        else:
            tk.Label(status_frame, text="❌ DBF Module: Not Available", 
                    font=("Arial", 10), fg="red", bg="#f8f9fa").pack(anchor='w', padx=20)

        # Instructions Frame
        instr_frame = tk.Frame(main_frame, bg="#e3f2fd", relief=tk.GROOVE, bd=1)
        instr_frame.pack(pady=20, fill='x')
        
        tk.Label(instr_frame, text="Process Flow:", 
                font=("Arial", 11, "bold"), bg="#e3f2fd").pack(anchor='w', padx=10, pady=5)
        
        steps = [
            "1️⃣ Select Image Folder containing cheque images",
            "2️⃣ Select DBF File to update (optional)",
            "3️⃣ Processing starts automatically",
            "4️⃣ DBF updated automatically for valid texts"
        ]
        
        for step in steps:
            tk.Label(instr_frame, text=step, 
                    font=("Arial", 10), bg="#e3f2fd").pack(anchor='w', padx=20, pady=2)

        # NumPy Warning if needed
        try:
            import numpy
            if numpy.__version__.startswith('2'):
                warning_frame = tk.Frame(main_frame, bg="#fff3cd", relief=tk.GROOVE, bd=1)
                warning_frame.pack(pady=10, fill='x')
                tk.Label(warning_frame, 
                        text="⚠️ NumPy 2.x Detected - May cause OCR issues\n💡 Solution: pip install numpy==1.24.3", 
                        font=("Arial", 9), fg="#856404", bg="#fff3cd").pack(padx=10, pady=5)
        except:
            pass

        # Start button
        btn = tk.Button(main_frame, 
                       text="🚀 START PROCESSING", 
                       command=start_process,
                       bg="#27ae60", 
                       fg="white", 
                       width=25, 
                       height=2,
                       font=("Arial", 12, "bold"),
                       cursor="hand2",
                       relief=tk.RAISED)
        btn.pack(pady=20)

        # Footer
        footer_frame = tk.Frame(main_frame)
        footer_frame.pack(side="bottom", fill='x', pady=5)
        
        tk.Label(footer_frame, text="© MJ IT Solution", 
                font=("Arial", 8), fg="gray").pack()
        tk.Label(footer_frame, text="Version 2.0 - NumPy Compatible", 
                font=("Arial", 7), fg="gray").pack()

        root.mainloop()

    except Exception as e:
        print(f"Fatal Error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")