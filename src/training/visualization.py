#!/usr/bin/env python3
"""
Training Visualization Module for HaWoR
Provides comprehensive visualization tools to verify training progress
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime

# Set matplotlib backend for Mac
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class TrainingVisualizer:
    """Visualization tools for HaWoR training verification"""

    def __init__(self, output_dir: str = "outputs"):
        """Initialize visualizer"""
        self.output_dir = Path(output_dir)
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")

        print(f"📊 Training Visualizer initialized: {self.vis_dir}")

    def plot_training_metrics(self, metrics: Dict, save_name: str = "training_metrics.png"):
        """Plot training metrics (loss curves, accuracy, etc.)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('HaWoR Training Progress', fontsize=16, fontweight='bold')

        epochs = list(range(len(metrics.get('train_loss', []))))

        # Loss subplot
        ax1 = axes[0, 0]
        ax1.plot(epochs, metrics.get('train_loss', []), label='Train Loss', linewidth=2)
        ax1.plot(epochs, metrics.get('val_loss', []), label='Val Loss', linewidth=2)
        ax1.set_title('Loss Curves', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Keypoint error subplot
        ax2 = axes[0, 1]
        ax2.plot(epochs, metrics.get('train_keypoint_error', []), label='Train Keypoint Error', linewidth=2)
        ax2.plot(epochs, metrics.get('val_keypoint_error', []), label='Val Keypoint Error', linewidth=2)
        ax2.set_title('Keypoint Error', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Error (mm)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning rate subplot
        ax3 = axes[1, 0]
        ax3.plot(epochs, metrics.get('learning_rate', []), linewidth=2, color='orange')
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # Combined metrics subplot
        ax4 = axes[1, 1]
        if metrics.get('train_pose_error'):
            ax4.plot(epochs, metrics.get('train_pose_error', []), label='Train Pose Error', linewidth=2)
        if metrics.get('val_pose_error'):
            ax4.plot(epochs, metrics.get('val_pose_error', []), label='Val Pose Error', linewidth=2)
        if metrics.get('train_shape_error'):
            ax4.plot(epochs, metrics.get('train_shape_error', []), label='Train Shape Error', linewidth=2)
        if metrics.get('val_shape_error'):
            ax4.plot(epochs, metrics.get('val_shape_error', []), label='Val Shape Error', linewidth=2)
        ax4.set_title('Additional Metrics', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Error')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.vis_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training metrics plot saved: {self.vis_dir / save_name}")

    def plot_sample_predictions(self, predictions: List[Dict], save_name: str = "sample_predictions.png"):
        """Plot sample predictions vs ground truth"""
        if not predictions:
            print("⚠️  No predictions to visualize")
            return

        n_samples = min(4, len(predictions))  # Show up to 4 samples
        fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
        fig.suptitle('Sample Predictions vs Ground Truth', fontsize=14, fontweight='bold')

        for i in range(n_samples):
            sample = predictions[i]

            # Prediction subplot
            if sample.get('pred_keypoints_2d') is not None:
                ax_pred = axes[0, i]
                ax_pred.scatter(sample['pred_keypoints_2d'][:, 0], sample['pred_keypoints_2d'][:, 1],
                              c='red', s=50, label='Predicted', alpha=0.7)
                ax_pred.set_title(f'Sample {i+1} - Predicted')
                ax_pred.invert_yaxis()
                ax_pred.legend()

            # Ground truth subplot
            if sample.get('gt_keypoints_2d') is not None:
                ax_gt = axes[1, i]
                ax_gt.scatter(sample['gt_keypoints_2d'][:, 0], sample['gt_keypoints_2d'][:, 1],
                            c='blue', s=50, label='Ground Truth', alpha=0.7)
                ax_gt.set_title(f'Sample {i+1} - Ground Truth')
                ax_gt.invert_yaxis()
                ax_gt.legend()

        plt.tight_layout()
        plt.savefig(self.vis_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Sample predictions plot saved: {self.vis_dir / save_name}")

    def plot_training_summary(self, final_metrics: Dict, save_name: str = "training_summary.png"):
        """Create a comprehensive training summary visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('HaWoR Training Summary', fontsize=16, fontweight='bold')

        # Final metrics bar chart
        ax1 = axes[0]
        metrics_names = ['Keypoint Error', 'Pose Error', 'Shape Error', 'Global Orient Error']
        final_values = [
            final_metrics.get('final_keypoint_error', 0),
            final_metrics.get('final_pose_error', 0),
            final_metrics.get('final_shape_error', 0),
            final_metrics.get('final_global_orient_error', 0)
        ]

        bars = ax1.bar(metrics_names, final_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax1.set_title('Final Error Metrics', fontweight='bold')
        ax1.set_ylabel('Error Value')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, final_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Training time breakdown
        ax2 = axes[1]
        phases = ['Data Loading', 'Training', 'Validation', 'Visualization']
        times = [1, 5, 1, 0.5]  # Placeholder values
        ax2.pie(times, labels=phases, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Training Time Breakdown', fontweight='bold')

        # Improvement over epochs
        ax3 = axes[2]
        epochs = list(range(1, 6))  # Assuming 5 epochs
        improvements = [0.8, 0.6, 0.4, 0.3, 0.25]  # Placeholder improvement curve
        ax3.plot(epochs, improvements, 'g-o', linewidth=3, markersize=8)
        ax3.set_title('Error Improvement Over Time', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Average Error')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.vis_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training summary plot saved: {self.vis_dir / save_name}")

    def create_training_report(self, metrics: Dict, config: Dict, save_name: str = "training_report.json"):
        """Create a comprehensive training report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": "HaWoR",
            "dataset": "ARCTIC",
            "configuration": config,
            "final_metrics": {
                "best_val_loss": min(metrics.get('val_loss', [0])),
                "final_train_loss": metrics.get('train_loss', [-1]),
                "final_val_loss": metrics.get('val_loss', [-1]),
                "keypoint_error_improvement": self._calculate_improvement(metrics.get('val_keypoint_error', [])),
                "training_stability": self._assess_stability(metrics.get('val_loss', []))
            },
            "training_summary": {
                "total_epochs": len(metrics.get('train_loss', [])),
                "device_used": metrics.get('device', 'unknown'),
                "convergence_epoch": self._find_convergence_epoch(metrics.get('val_loss', [])),
                "best_epoch": metrics.get('val_loss', []).index(min(metrics.get('val_loss', []))) + 1 if metrics.get('val_loss') else 0
            },
            "recommendations": self._generate_recommendations(metrics, config)
        }

        # Save report
        report_path = self.vis_dir / save_name
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Training report saved: {report_path}")
        return report

    def _calculate_improvement(self, errors: List[float]) -> float:
        """Calculate improvement percentage"""
        if len(errors) < 2:
            return 0.0
        return ((errors[0] - errors[-1]) / errors[0]) * 100

    def _assess_stability(self, losses: List[float]) -> str:
        """Assess training stability"""
        if len(losses) < 3:
            return "Insufficient data"
        std = np.std(losses)
        if std < 0.01:
            return "Very Stable"
        elif std < 0.05:
            return "Stable"
        elif std < 0.1:
            return "Moderately Stable"
        else:
            return "Unstable"

    def _find_convergence_epoch(self, losses: List[float]) -> int:
        """Find epoch where training converged"""
        if len(losses) < 3:
            return len(losses)
        for i in range(2, len(losses)):
            if abs(losses[i] - losses[i-1]) < 0.001 and abs(losses[i-1] - losses[i-2]) < 0.001:
                return i + 1
        return len(losses)

    def _generate_recommendations(self, metrics: Dict, config: Dict) -> List[str]:
        """Generate training recommendations"""
        recommendations = []

        # Learning rate recommendations
        if metrics.get('val_loss') and len(metrics['val_loss']) > 1:
            if metrics['val_loss'][-1] > metrics['val_loss'][0]:
                recommendations.append("Consider reducing learning rate - validation loss is increasing")

        # Batch size recommendations
        if config.get('batch_size', 1) == 1:
            recommendations.append("Consider increasing batch size for more stable training")

        # Early stopping recommendations
        if len(metrics.get('val_loss', [])) < 3:
            recommendations.append("Consider training for more epochs")

        return recommendations or ["Training configuration looks good!"]

def create_quick_visualization_check(output_dir: str = "outputs"):
    """Create a quick visualization to verify everything works"""
    print("🧪 Creating quick visualization test...")

    vis = TrainingVisualizer(output_dir)

    # Create sample data
    sample_metrics = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25],
        'val_loss': [0.9, 0.7, 0.5, 0.35, 0.3],
        'train_keypoint_error': [15.0, 12.0, 9.0, 7.0, 6.0],
        'val_keypoint_error': [16.0, 13.0, 10.0, 8.0, 7.0],
        'learning_rate': [1e-5, 9e-6, 8e-6, 7e-6, 5e-6]
    }

    # Generate visualizations
    vis.plot_training_metrics(sample_metrics, "test_metrics.png")
    vis.plot_training_summary({
        'final_keypoint_error': 7.0,
        'final_pose_error': 0.3,
        'final_shape_error': 0.05,
        'final_global_orient_error': 0.1
    }, "test_summary.png")

    vis.create_training_report(sample_metrics, {}, "test_report.json")

    print("✅ Quick visualization test completed!")
    print(f"📁 Check: {vis.vis_dir}")

if __name__ == "__main__":
    create_quick_visualization_check()
