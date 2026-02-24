# batch_process.py - Fixed version with absolute Python path

import subprocess
import shutil
import sys
from pathlib import Path

# Get the absolute path to the current Python executable
python_exe = sys.executable
print(f"🔧 Using Python: {python_exe}")

# Get all images
all_images = list(Path("cheque-images").glob("*.tiff"))
total = len(all_images)
print(f"📊 Total images: {total}")

# Process in batches of 100
batch_size = 100

for i in range(0, total, batch_size):
    batch_num = i // batch_size + 1
    total_batches = (total // batch_size) + 1
    
    print(f"\n{'='*50}")
    print(f"📦 BATCH {batch_num}/{total_batches}")
    print(f"📸 Images {i+1} to {min(i+batch_size, total)}")
    print('='*50)
    
    # Create temp folder for this batch
    temp_dir = Path(f"temp_batch_{batch_num}")
    temp_dir.mkdir(exist_ok=True)
    
    # COPY files to temp folder
    batch_files = all_images[i:i+batch_size]
    for f in batch_files:
        shutil.copy2(f, temp_dir / f.name)
        print(f"  Copying: {f.name}")
    
    print(f"✅ Copied {len(batch_files)} files to {temp_dir}")
    
    # Use the SAME Python executable with full path
    cmd = [
        python_exe, "run_production.py",  # Use sys.executable instead of "python"
        "--input-dir", str(temp_dir),
        "--output-dir", f"results/batch_{batch_num}",
        "--processes-per-cpu", "2",
        "--threads-per-process", "1"
    ]
    
    print(f"🚀 Processing batch {batch_num}...")
    print(f"📋 Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"❌ Batch {batch_num} failed with code {result.returncode}")
        # Don't stop - try next batch anyway
        # break
    
    # Clean up temp folder
    shutil.rmtree(temp_dir)
    print(f"✅ Batch {batch_num} complete. Cleaned up temp folder.")

print("\n🎉 ALL BATCHES COMPLETE!")

# Combine all results
print("\n📁 Combining results...")
all_results = []
for batch_dir in Path("results").glob("batch_*"):
    csv_files = list(batch_dir.glob("FINAL_OUTPUT_*.csv"))
    if csv_files:
        all_results.extend(csv_files)

if all_results:
    # Create combined file
    with open("results/all_results.csv", "w", encoding="utf-8") as outfile:
        # Write header from first file
        with open(all_results[0], "r", encoding="utf-8") as first:
            header = first.readline()
            outfile.write(header)
        
        # Write all data
        for csv_file in all_results:
            with open(csv_file, "r", encoding="utf-8") as infile:
                next(infile)  # Skip header
                outfile.write(infile.read())
    
    print(f"✅ Combined {len(all_results)} batch files into results/all_results.csv")
    print("\n📊 To open final result:")
    print("   notepad results/all_results.csv")
else:
    print("❌ No result files found!")