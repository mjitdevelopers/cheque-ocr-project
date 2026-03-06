# app_debug.py
import tkinter as tk
from tkinter import messagebox
import sys
import os
import traceback

# Write debug info to file
with open('debug_log.txt', 'w', encoding='utf-8') as f:
    f.write("="*50 + "\n")
    f.write("DEBUG INFORMATION\n")
    f.write("="*50 + "\n")
    f.write(f"Python Version: {sys.version}\n")
    f.write(f"Python Executable: {sys.executable}\n")
    f.write(f"Running as EXE: {getattr(sys, 'frozen', False)}\n")
    f.write(f"Current Directory: {os.getcwd()}\n")
    
    f.write("\n" + "="*50 + "\n")
    f.write("TRYING TO IMPORT PADDLEOCR\n")
    f.write("="*50 + "\n")
    
    try:
        f.write("Step 1: Importing paddleocr module...\n")
        import paddleocr
        f.write(f"✅ PaddleOCR module found at: {paddleocr.__file__}\n")
        
        f.write("\nStep 2: Setting environment variables...\n")
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['FLAGS_logtostderr'] = '0'
        f.write("✅ Environment variables set\n")
        
        f.write("\nStep 3: Importing PaddleOCR class...\n")
        from paddleocr import PaddleOCR
        f.write("✅ PaddleOCR class imported\n")
        
        f.write("\nStep 4: Initializing PaddleOCR...\n")
        ocr = PaddleOCR(use_angle_cls=False, show_log=False, lang='en')
        f.write("✅ PaddleOCR initialized successfully!\n")
        OCR_AVAILABLE = True
        
    except Exception as e:
        OCR_AVAILABLE = False
        f.write(f"\n❌ Error: {str(e)}\n")
        f.write("\nFull traceback:\n")
        traceback.print_exc(file=f)

# Create GUI
root = tk.Tk()
root.title("OCR Debug")
root.geometry("400x300")

status = "✅ OCR Ready" if OCR_AVAILABLE else "❌ OCR Failed"
tk.Label(root, text=status, font=("Arial", 14, "bold")).pack(pady=20)

def show_log():
    try:
        with open('debug_log.txt', 'r') as log_file:
            content = log_file.read()
            messagebox.showinfo("Debug Log", content)
    except:
        messagebox.showerror("Error", "Could not read debug_log.txt")

tk.Button(root, text="Show Log", command=show_log).pack(pady=10)
tk.Button(root, text="Exit", command=sys.exit).pack(pady=10)

root.mainloop()