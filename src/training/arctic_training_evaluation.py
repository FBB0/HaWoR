#!/usr/bin/env python3
"""
ARCTIC Training Evaluation System for HaWoR
Simplified evaluation system adapted for ARCTIC data
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List, Optional
import logging
from collections import defaultdict

# Add HaWoR to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hawor_interface import HaWoRInterface

class ArcticTrainingMetrics:
    """Simple metrics for ARCTIC training evaluation"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.losses = []
        self.accuracies = []
        self.epoch_times = []
        self.batch_times = []

    def update(self, loss: float, accuracy: float = 0.0):
        """Update metrics with new values"""
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def update_time(self, epoch_time: float = 0.0, batch_time: float = 0.0):
        """Update timing metrics"""
        if epoch_time > 0:
            self.epoch_times.append(epoch_time)
        if batch_time > 0:
            self.batch_times.append(batch_time)

    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        if not self.losses:
            return {"status": "No data"}

        return {
            "total_batches": len(self.losses),
            "avg_loss": np.mean(self.losses),
            "min_loss": np.min(self.losses),
            "max_loss": np.max(self.losses),
            "avg_accuracy": np.mean(self.accuracies) if self.accuracies else 0.0,
            "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0.0,
            "avg_batch_time": np.mean(self.batch_times) if self.batch_times else 0.0,
            "status": "Training completed"
        }

    def log_epoch_summary(self, epoch: int, total_epochs: int):
        """Log epoch summary"""
        if not self.losses:
            print(f"    📊 Epoch {epoch}/{total_epochs}: No metrics recorded")
            return

        summary = self.get_summary()
        print(f"    📊 Epoch {epoch}/{total_epochs} Summary:")
        print(f"      📈 Avg Loss: {summary['avg_loss']:.4f}")
        print(f"      🎯 Avg Accuracy: {summary['avg_accuracy']:.4f}")
        print(f"      ⏱️  Avg Epoch Time: {summary['avg_epoch_time']:.2f}s")

class ArcticTrainingEvaluator:
    """Evaluate HaWoR training on ARCTIC data"""

    def __init__(self, device: str = 'auto'):
        """Initialize evaluator"""
        self.device = device
        self.metrics = ArcticTrainingMetrics()
        print("  📊 ARCTIC Training Evaluator initialized")

    def evaluate_batch(self, batch_data: Dict, model_output: Dict) -> Dict:
        """Evaluate single batch"""
        # Simple evaluation metrics
        loss = model_output.get('loss', 0.0)
        accuracy = model_output.get('accuracy', 0.0)

        self.metrics.update(loss, accuracy)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'batch_size': len(batch_data.get('images', []))
        }

    def evaluate_epoch(self, epoch: int, total_epochs: int):
        """Evaluate epoch performance"""
        print(f"  📊 Evaluating epoch {epoch}/{total_epochs}...")

        # Get metrics summary
        summary = self.metrics.get_summary()

        if summary.get('status') == 'No data':
            print("    ⚠️  No metrics available for evaluation")
            return {}

        # Log evaluation results
        self.log_epoch_results(epoch, summary)

        return summary

    def log_epoch_results(self, epoch: int, summary: Dict):
        """Log epoch evaluation results"""
        print(f"  📈 Epoch {epoch} Evaluation Results:")
        print(f"    📊 Average Loss: {summary['avg_loss']:.4f}")
        print(f"    🎯 Average Accuracy: {summary['avg_accuracy']:.4f}")
        print(f"    ⏱️  Average Epoch Time: {summary['avg_epoch_time']:.2f}s")
        print(f"    📈 Total Batches: {summary['total_batches']}")

    def save_evaluation_results(self, output_dir: Path):
        """Save evaluation results to file"""
        try:
            results_file = output_dir / "arctic_evaluation_results.json"

            results = {
                "evaluation_summary": self.metrics.get_summary(),
                "device": self.device,
                "data_type": "ARCTIC",
                "timestamp": str(time.time())
            }

            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"  💾 Evaluation results saved to: {results_file}")
            return True

        except Exception as e:
            print(f"  ❌ Error saving evaluation results: {e}")
            return False

    def generate_training_report(self, output_dir: Path):
        """Generate comprehensive training report"""
        try:
            report_file = output_dir / "arctic_training_report.txt"

            with open(report_file, 'w') as f:
                f.write("HaWoR ARCTIC Training Report\n")
                f.write("=" * 40 + "\n")
                f.write(f"Device: {self.device}\n")
                f.write("Data: ARCTIC s01\n")
                f.write("Model: HaWoR Vision Transformer\n\n")

                summary = self.metrics.get_summary()
                f.write("Training Results:\n")
                f.write(f"- Total Batches: {summary['total_batches']}\n")
                f.write(f"- Average Loss: {summary['avg_loss']:.4f}\n")
                f.write(f"- Average Accuracy: {summary['avg_accuracy']:.4f}\n")
                f.write(f"- Average Epoch Time: {summary['avg_epoch_time']:.2f}s\n")
                f.write(f"- Status: {summary['status']}\n")

            print(f"  📄 Training report saved to: {report_file}")
            return True

        except Exception as e:
            print(f"  ❌ Error generating training report: {e}")
            return False

def create_arctic_evaluator(config: Dict) -> ArcticTrainingEvaluator:
    """Create ARCTIC training evaluator"""
    device = config.get('hardware', {}).get('accelerator', 'auto')
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    return ArcticTrainingEvaluator(device=device)
