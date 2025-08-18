import subprocess
import sys
import time

script = "imusim_SMPL.py"

while True:
    print("Launching", script)
    result = subprocess.run([sys.executable, script])
    if result.returncode == 0:
        print("Finished successfully.")
        break
    else:
        print(f"Script crashed (exit code {result.returncode}), restarting in 2s...")
        time.sleep(2)
