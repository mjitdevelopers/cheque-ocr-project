# fast_run_4core.py - For 1 CPU with 4 cores

import os
import subprocess
import psutil

# Step 1: Clear environment
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# Step 2: Check your cores
cores = psutil.cpu_count(logical=True)  # This will show 8 if hyperthreading is on
physical = psutil.cpu_count(logical=False)  # This will show 4

print(f"✅ Physical cores: {physical}")
print(f"✅ Logical threads: {cores}")

# Step 3: For 4-core CPU - best settings
if cores >= 8:  # With hyperthreading
    processes = 3
    threads = 2
else:  # Without hyperthreading
    processes = 2
    threads = 2

print(f"⚡ Using {processes} processes with {threads} threads each")

# Step 4: Run with correct settings
cmd = [
    "python", "run_production.py",
    "--input-dir", "cheque-images",
    "--num-cpus", "1",                    # IMPORTANT: 1 CPU, not 4
    "--processes-per-cpu", str(processes),
    "--threads-per-process", str(threads)
]

print(f"\n🚀 Running: {' '.join(cmd)}")
subprocess.run(cmd)