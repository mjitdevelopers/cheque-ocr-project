# test_4core.py - Find best settings for YOUR 4-core CPU

import subprocess
import time

# Test different settings for 4-core CPU
settings = [
    {"name": "Safe", "procs": 2, "threads": 2},     # 2 processes × 2 threads = 4 threads
    {"name": "Medium", "procs": 3, "threads": 2},   # 3 processes × 2 threads = 6 threads
    {"name": "Aggressive", "procs": 4, "threads": 1}, # 4 processes × 1 thread = 4 threads
    {"name": "Hyper", "procs": 4, "threads": 2},    # 4 processes × 2 threads = 8 threads (if you have hyperthreading)
]

print("="*60)
print("🔬 TESTING BEST SETTINGS FOR YOUR 4-CORE CPU")
print("="*60)

best_speed = 0
best_setting = None

for setting in settings:
    print(f"\n📊 Testing: {setting['name']}")
    print(f"   {setting['procs']} processes × {setting['threads']} threads = {setting['procs'] * setting['threads']} total threads")
    
    # Test with 5 images
    cmd = [
        "python", "run_production.py",
        "--input-dir", "cheque-images",
        "--num-cpus", "1",
        "--processes-per-cpu", str(setting['procs']),
        "--threads-per-process", str(setting['threads']),
        "--limit", "5"
    ]
    
    start = time.time()
    subprocess.run(cmd)
    elapsed = time.time() - start
    
    speed = 5 / (elapsed / 60)  # images per minute
    print(f"⏱️  Time for 5 images: {elapsed:.1f} seconds")
    print(f"⚡ Speed: {speed:.1f} images/minute")
    
    if speed > best_speed:
        best_speed = speed
        best_setting = setting

print("\n" + "="*60)
print(f"✅ BEST SETTING FOR YOUR PC:")
print(f"   {best_setting['name']}: {best_setting['procs']} processes, {best_setting['threads']} threads")
print(f"   Speed: {best_speed:.1f} images/minute")
print("="*60)

print("\n📝 To run full production with best settings:")
print(f"python run_production.py --input-dir cheque-images --num-cpus 1 --processes-per-cpu {best_setting['procs']} --threads-per-process {best_setting['threads']}")