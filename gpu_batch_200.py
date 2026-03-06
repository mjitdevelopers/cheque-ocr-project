# gpu_batch_200.py - Process 200 images per batch with GPU

import subprocess
import sys
import shutil
from pathlib import Path

print("="*60)
print("🚀 GPU CHEQUE OCR - 200 IMAGES PER BATCH")
print("="*60)

# Get all images
all_images = list(Path("cheque-images").glob("*.tiff")) + list(Path("cheque-images").glob("*.tif"))
total = len(all_images)
print(f"📊 Total images: {total}")

# Process in batches of 200
batch_size = 200

for i in range(0, total, batch_size):
    batch_num = i // batch_size + 1
    total_batches = (total + batch_size - 1) // batch_size
    
    print(f"\n{'='*50}")
    print(f"📦 BATCH {batch_num}/{total_batches}")
    print(f"📸 Images {i+1} to {min(i+batch_size, total)}")
    print('='*50)
    
    # Create temp folder
    temp_dir = Path(f"gpu_batch_{batch_num}")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy files
    batch_files = all_images[i:i+batch_size]
    print(f"📋 Copying {len(batch_files)} files...")
    for idx, f in enumerate(batch_files):
        shutil.copy2(f, temp_dir / f.name)
        if (idx + 1) % 50 == 0:
            print(f"   Copied {idx+1}/{len(batch_files)}")
    
    print(f"✅ Copied {len(batch_files)} files to {temp_dir}")
    
    # Process with GPU
    cmd = [
        sys.executable, "run_production.py",
        "--input-dir", str(temp_dir),
        "--output-dir", f"gpu_results/batch_{batch_num}",
        "--processes-per-cpu", "2",
        "--threads-per-process", "1",
        "--use-gpu",
        "--gpu-id", "0"
    ]
    
    print(f"🎮 Processing batch {batch_num} with GPU...")
    print(f"📋 Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"⚠️ Batch {batch_num} had issues (code: {result.returncode}), but continuing...")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print(f"✅ Batch {batch_num} complete. Cleaned up temp folder.")

print("\n" + "="*60)
print("🎉 ALL BATCHES COMPLETE!")
print("="*60)

# Combine all results
print("\n📁 Combining payee names...")

all_payees = []
for batch_dir in Path("gpu_results").glob("batch_*"):
    csv_files = list(batch_dir.glob("FINAL_PAYEE_NAMES_*.csv"))
    if csv_files:
        all_payees.extend(csv_files)
    else:
        # Try alternate naming
        csv_files = list(batch_dir.glob("*.csv"))
        if csv_files:
            all_payees.extend(csv_files)

if all_payees:
    # Create combined file
    output_file = "gpu_results/ALL_PAYEE_NAMES.csv"
    with open(output_file, "w", encoding="utf-8") as outfile:
        # Write header
        with open(all_payees[0], "r", encoding="utf-8") as first:
            header = first.readline()
            if 'Payee Name' in header:
                outfile.write(header)
            else:
                outfile.write("Payee Name,Filename\n")
        
        # Write all data
        total_payees = 0
        for csv_file in all_payees:
            with open(csv_file, "r", encoding="utf-8") as infile:
                lines = infile.readlines()
                if len(lines) > 1:  # Has header + data
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            outfile.write(line)
                            total_payees += 1
        
        print(f"✅ Combined {total_payees} payee names")
    
    print("\n📊 FINAL OUTPUT:")
    print(f"   📁 {output_file}")
    print("\n📋 SAMPLE (first 10 lines):")
    print("-"*40)
    with open(output_file, "r") as f:
        for i, line in enumerate(f):
            if i < 10:
                print(f"   {line.strip()}")
    print("-"*40)
else:
    print("❌ No result files found!")
    
print("\n✅ DONE! Check gpu_results/ALL_PAYEE_NAMES.csv")