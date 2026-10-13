
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 5: Production Data Pipeline")
    print("[INFO] =========================================")

    scripts = [
        "validate_data.py",
        "preprocess_pipeline.py",
        "validate_outputs.py",
    ]

    for script in scripts:
        script_path = PROJECT_DIR / script

        if not script_path.exists():
            print(f"[ERROR] Required script not found: {script_path}")
            sys.exit(1)

        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_DIR
        )

        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted at {script}.")
            sys.exit(result.returncode)

    print("\n[SUCCESS] Production pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
