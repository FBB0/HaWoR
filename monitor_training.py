#!/usr/bin/env python3
"""
HaWoR Training Monitoring Dashboard
Simple monitoring for training progress
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_training(log_dir: str = "./training_logs"):
    """Monitor training progress"""
    log_path = Path(log_dir)

    print("📊 HaWoR Training Monitor")
    print("=" * 40)
    print("Press Ctrl+C to stop monitoring\n")

    try:
        while True:
            # Check for latest metrics
            metrics_files = list(log_path.glob("**/metrics.json"))

            if metrics_files:
                latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)

                try:
                    with open(latest_file, 'r') as f:
                        metrics = json.load(f)

                    print(f"\r🕐 {datetime.now().strftime('%H:%M:%S')} | "
                          f"Epoch: {metrics.get('epoch', 'N/A')} | "
                          f"Loss: {metrics.get('train_loss', 'N/A'):.4f} | "
                          f"Val Loss: {metrics.get('val_loss', 'N/A'):.4f}", end="")

                except Exception:
                    print(f"\r🕐 {datetime.now().strftime('%H:%M:%S')} | "
                          f"Monitoring... (no metrics yet)", end="")
            else:
                print(f"\r🕐 {datetime.now().strftime('%H:%M:%S')} | "
                      f"Waiting for training to start...", end="")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n📊 Monitoring stopped")

if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "./training_logs"
    monitor_training(log_dir)
