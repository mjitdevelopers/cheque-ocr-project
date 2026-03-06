"""
EXE Builder for Issuer Processor V2
Run this to create a standalone executable
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    print("="*60)
    print("Issuer Processor V2 - EXE Builder")
    print("="*60)
    
    # Check if source file exists
    if not Path("issuer_processor_v2.py").exists():
        print("❌ ERROR: issuer_processor_v2.py not found!")
        return
    
    # Install required packages
    print("\n1. Installing required packages...")
    packages = [
        "pyinstaller",
        "paddleocr",
        "paddlepaddle",
        "opencv-python",
        "numpy",
        "pandas",
        "dbfread",
        "dbf"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True)
    
    # Clean previous builds
    print("\n2. Cleaning previous builds...")
    for folder in ['build', 'dist', '__pycache__']:
        if Path(folder).exists():
            shutil.rmtree(folder)
    
    # Create the EXE
    print("\n3. Creating executable (this may take a few minutes)...")
    
    cmd = [
        "pyinstaller",
        "--onefile",              # Single executable file
        "--console",              # Show console window
        "--name=IssuerProcessor", # Name of the exe
        "--clean",                 # Clean cache
        "--add-data=issuer_processor_v2.py;.",  # Include source
        "--hidden-import=paddleocr",
        "--hidden-import=paddle",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=pandas",
        "--hidden-import=dbfread",
        "--hidden-import=dbf",
        "issuer_processor_v2.py"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ EXE CREATED SUCCESSFULLY!")
        print("="*60)
        print(f"📍 Location: dist/IssuerProcessor.exe")
        print(f"📍 Size: {Path('dist/IssuerProcessor.exe').stat().st_size / 1024 / 1024:.1f} MB")
        print("\n📋 Instructions for client:")
        print("   1. Double-click IssuerProcessor.exe")
        print("   2. Follow the prompts")
        print("   3. Output files will be created in the same folder")
        print("="*60)
    else:
        print("❌ Failed to create EXE. Check errors above.")

if __name__ == "__main__":
    main()