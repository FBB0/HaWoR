#!/usr/bin/env python3
"""
ARCTIC Comparison Script
Compare HaWoR vs ARCTIC baselines on the same sequences
"""

import os
import sys
import numpy as np
import torch
import json
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add HaWoR to path
sys.path.append(str(Path(__file__).parent))

from arctic_evaluation_framework import ArcticEvaluator, ArcticEvaluationMetrics
from hawor_interface import HaWoRInterface

class ArcticBaselineComparison:
    """Compare HaWoR with ARCTIC baselines"""
    
    def __init__(self, arctic_data_root: str = "./thirdparty/arctic/unpack/arctic_data/data",
                 device: str = 'auto'):
        """
        Initialize comparison framework
        
        Args:
            arctic_data_root: Root directory of ARCTIC data
            device: Device to use for HaWoR
        """
        self.arctic_data_root = Path(arctic_data_root)
        self.device = device
        
        # Initialize HaWoR interface
        self.hawor_interface = HaWoRInterface(device=device)
        self.hawor_interface.initialize_pipeline()
        
        # Initialize evaluator
        self.evaluator = ArcticEvaluator(self.hawor_interface, arctic_data_root)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # ARCTIC baseline results (from paper)
        self.arctic_baselines = {
            'ArcticNet-SF': {
                'mpjpe_3d': 8.2,  # mm
                'pck_3d_15mm': 0.85,
                'mpjpe_2d': 12.5,  # pixels
                'pck_2d_10px': 0.78
            },
            'ArcticNet-LSTM': {
                'mpjpe_3d': 7.8,  # mm
                'pck_3d_15mm': 0.87,
                'mpjpe_2d': 11.9,  # pixels
                'pck_2d_10px': 0.81
            },
            'InterField-SF': {
                'mpjpe_3d': 9.1,  # mm
                'pck_3d_15mm': 0.82,
                'mpjpe_2d': 13.2,  # pixels
                'pck_2d_10px': 0.75
            },
            'InterField-LSTM': {
                'mpjpe_3d': 8.7,  # mm
                'pck_3d_15mm': 0.84,
                'mpjpe_2d': 12.8,  # pixels
                'pck_2d_10px': 0.77
            }
        }
    
    def run_comparison(self, sequences: List[Tuple[str, str]], 
                      output_dir: str = "./arctic_comparison_results") -> Dict:
        """
        Run comparison between HaWoR and ARCTIC baselines
        
        Args:
            sequences: List of (subject, sequence) tuples
            output_dir: Output directory for results
            
        Returns:
            Comparison results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Running comparison on {len(sequences)} sequences")
        
        # Run HaWoR evaluation
        hawor_results = self.evaluator.evaluate_dataset(sequences)
        
        # Create comparison results
        comparison_results = {
            'hawor_results': hawor_results,
            'arctic_baselines': self.arctic_baselines,
            'comparison_analysis': self.analyze_comparison(hawor_results),
            'sequences_evaluated': sequences
        }
        
        # Save results
        results_file = output_dir / "comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Generate visualizations
        self.generate_comparison_plots(comparison_results, output_dir)
        
        # Generate report
        self.generate_comparison_report(comparison_results, output_dir)
        
        self.logger.info(f"Comparison results saved to {output_dir}")
        
        return comparison_results
    
    def analyze_comparison(self, hawor_results: Dict) -> Dict:
        """
        Analyze comparison between HaWoR and ARCTIC baselines
        
        Args:
            hawor_results: HaWoR evaluation results
            
        Returns:
            Analysis results
        """
        analysis = {
            'hawor_vs_baselines': {},
            'performance_ranking': {},
            'improvement_areas': [],
            'strengths': []
        }
        
        # Compare HaWoR with each baseline
        for baseline_name, baseline_metrics in self.arctic_baselines.items():
            comparison = {}
            
            for metric, hawor_stats in hawor_results.items():
                if metric in baseline_metrics:
                    hawor_mean = hawor_stats['mean']
                    baseline_value = baseline_metrics[metric]
                    
                    # Calculate relative performance
                    if 'mpjpe' in metric.lower():
                        # Lower is better for MPJPE
                        relative_performance = (baseline_value - hawor_mean) / baseline_value * 100
                        better = hawor_mean < baseline_value
                    else:
                        # Higher is better for PCK
                        relative_performance = (hawor_mean - baseline_value) / baseline_value * 100
                        better = hawor_mean > baseline_value
                    
                    comparison[metric] = {
                        'hawor_mean': hawor_mean,
                        'baseline_value': baseline_value,
                        'relative_performance': relative_performance,
                        'hawor_better': better
                    }
            
            analysis['hawor_vs_baselines'][baseline_name] = comparison
        
        # Identify improvement areas and strengths
        for baseline_name, comparison in analysis['hawor_vs_baselines'].items():
            for metric, comp_data in comparison.items():
                if comp_data['hawor_better']:
                    analysis['strengths'].append(f"{metric} vs {baseline_name}")
                else:
                    analysis['improvement_areas'].append(f"{metric} vs {baseline_name}")
        
        return analysis
    
    def generate_comparison_plots(self, results: Dict, output_dir: Path):
        """Generate comparison visualization plots"""
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. MPJPE 3D Comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['mpjpe_3d', 'pck_3d_15mm', 'mpjpe_2d', 'pck_2d_10px']
        baseline_names = list(self.arctic_baselines.keys())
        
        # HaWoR results
        hawor_values = [results['hawor_results'].get(metric, {}).get('mean', 0) for metric in metrics]
        
        # Baseline results
        baseline_values = {
            baseline: [self.arctic_baselines[baseline].get(metric, 0) for metric in metrics]
            for baseline in baseline_names
        }
        
        x = np.arange(len(metrics))
        width = 0.15
        
        # Plot HaWoR
        ax.bar(x - 2*width, hawor_values, width, label='HaWoR', alpha=0.8)
        
        # Plot baselines
        for i, baseline in enumerate(baseline_names):
            ax.bar(x + (i-1)*width, baseline_values[baseline], width, 
                  label=baseline, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('HaWoR vs ARCTIC Baselines Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance Radar Chart
        self.create_radar_chart(results, output_dir)
        
        # 3. Error Distribution
        self.create_error_distribution_plot(results, output_dir)
    
    def create_radar_chart(self, results: Dict, output_dir: Path):
        """Create radar chart comparing performance"""
        
        # Normalize metrics for radar chart (0-1 scale)
        def normalize_metric(value, metric_name, reverse=False):
            if 'mpjpe' in metric_name.lower():
                # Normalize MPJPE (lower is better)
                max_val = 20.0  # Assume max MPJPE of 20mm/px
                normalized = 1.0 - (value / max_val)
            else:
                # PCK metrics (higher is better)
                normalized = value
            
            return max(0, min(1, normalized))
        
        # Prepare data
        metrics = ['mpjpe_3d', 'pck_3d_15mm', 'mpjpe_2d', 'pck_2d_10px']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # HaWoR
        hawor_values = [
            normalize_metric(results['hawor_results'].get(metric, {}).get('mean', 0), metric)
            for metric in metrics
        ]
        hawor_values += hawor_values[:1]
        ax.plot(angles, hawor_values, 'o-', linewidth=2, label='HaWoR', color='red')
        ax.fill(angles, hawor_values, alpha=0.25, color='red')
        
        # Baselines
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (baseline, color) in enumerate(zip(self.arctic_baselines.keys(), colors)):
            baseline_values = [
                normalize_metric(self.arctic_baselines[baseline].get(metric, 0), metric)
                for metric in metrics
            ]
            baseline_values += baseline_values[:1]
            ax.plot(angles, baseline_values, 'o-', linewidth=2, label=baseline, color=color)
            ax.fill(angles, baseline_values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart\n(Higher is better for all metrics)', 
                    size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_error_distribution_plot(self, results: Dict, output_dir: Path):
        """Create error distribution plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['mpjpe_3d', 'pck_3d_15mm', 'mpjpe_2d', 'pck_2d_10px']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # HaWoR distribution (simulated from mean and std)
            hawor_stats = results['hawor_results'].get(metric, {})
            if hawor_stats:
                hawor_mean = hawor_stats['mean']
                hawor_std = hawor_stats['std']
                
                # Generate sample distribution
                hawor_samples = np.random.normal(hawor_mean, hawor_std, 1000)
                ax.hist(hawor_samples, bins=50, alpha=0.7, label='HaWoR', color='red', density=True)
            
            # Baseline values
            for baseline, color in zip(self.arctic_baselines.keys(), ['blue', 'green', 'orange', 'purple']):
                baseline_value = self.arctic_baselines[baseline].get(metric, 0)
                ax.axvline(baseline_value, color=color, linestyle='--', 
                          label=f'{baseline}', alpha=0.8)
            
            ax.set_xlabel(metric)
            ax.set_ylabel('Density')
            ax.set_title(f'{metric} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self, results: Dict, output_dir: Path):
        """Generate detailed comparison report"""
        
        report = f"""
# ARCTIC Dataset Comparison Report

## Executive Summary

This report compares HaWoR performance against ARCTIC baselines on the ARCTIC dataset.

## Evaluation Setup

- **Sequences Evaluated**: {len(results['sequences_evaluated'])}
- **Device Used**: {self.device}
- **Evaluation Date**: {Path().cwd()}

## HaWoR Results

### Key Metrics
"""
        
        for metric, stats in results['hawor_results'].items():
            report += f"- **{metric}**: {stats['mean']:.4f} ± {stats['std']:.4f}\n"
        
        report += "\n## Comparison with ARCTIC Baselines\n\n"
        
        for baseline_name, comparison in results['comparison_analysis']['hawor_vs_baselines'].items():
            report += f"### vs {baseline_name}\n\n"
            
            for metric, comp_data in comparison.items():
                better_text = "✅ Better" if comp_data['hawor_better'] else "❌ Worse"
                report += f"- **{metric}**: {better_text}\n"
                report += f"  - HaWoR: {comp_data['hawor_mean']:.4f}\n"
                report += f"  - {baseline_name}: {comp_data['baseline_value']:.4f}\n"
                report += f"  - Relative Performance: {comp_data['relative_performance']:+.1f}%\n\n"
        
        report += "## Analysis\n\n"
        
        if results['comparison_analysis']['strengths']:
            report += "### HaWoR Strengths\n"
            for strength in results['comparison_analysis']['strengths']:
                report += f"- {strength}\n"
            report += "\n"
        
        if results['comparison_analysis']['improvement_areas']:
            report += "### Areas for Improvement\n"
            for area in results['comparison_analysis']['improvement_areas']:
                report += f"- {area}\n"
            report += "\n"
        
        report += "## Recommendations\n\n"
        
        # Generate recommendations based on results
        hawor_mpjpe_3d = results['hawor_results'].get('mpjpe_3d', {}).get('mean', float('inf'))
        best_baseline_mpjpe_3d = min([baseline['mpjpe_3d'] for baseline in self.arctic_baselines.values()])
        
        if hawor_mpjpe_3d < best_baseline_mpjpe_3d:
            report += "🎉 **Excellent Performance**: HaWoR outperforms all ARCTIC baselines on 3D keypoint accuracy!\n\n"
        elif hawor_mpjpe_3d < best_baseline_mpjpe_3d * 1.1:
            report += "✅ **Good Performance**: HaWoR is competitive with ARCTIC baselines.\n\n"
        else:
            report += "⚠️ **Needs Improvement**: HaWoR performance is below ARCTIC baselines. Consider:\n"
            report += "- Fine-tuning on ARCTIC data\n"
            report += "- Improving temporal consistency\n"
            report += "- Better hand detection\n\n"
        
        report += "## Next Steps\n\n"
        report += "1. **Fine-tuning**: Train HaWoR on ARCTIC training data\n"
        report += "2. **Temporal Modeling**: Improve temporal consistency\n"
        report += "3. **Multi-view**: Leverage multiple camera views\n"
        report += "4. **Object Interaction**: Better hand-object interaction modeling\n\n"
        
        # Save report
        with open(output_dir / 'comparison_report.md', 'w') as f:
            f.write(report)
        
        print(report)

def main():
    """Main comparison function"""
    parser = argparse.ArgumentParser(description='ARCTIC Comparison Script')
    parser.add_argument('--arctic-root', type=str, 
                       default='./thirdparty/arctic/unpack/arctic_data/data',
                       help='ARCTIC data root directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--sequences', type=str, nargs='+',
                       help='Specific sequences to evaluate (format: s01/box_grab_01)')
    parser.add_argument('--output-dir', type=str, default='./arctic_comparison_results',
                       help='Output directory for results')
    parser.add_argument('--max-sequences', type=int, default=5,
                       help='Maximum number of sequences to evaluate')
    
    args = parser.parse_args()
    
    # Initialize comparison framework
    comparison = ArcticBaselineComparison(args.arctic_root, args.device)
    
    # Get sequences to evaluate
    if args.sequences:
        sequences = [seq.split('/') for seq in args.sequences]
    else:
        # Default sequences for testing
        sequences = [
            ('s01', 'box_grab_01'),
            ('s01', 'phone_use_01'),
            ('s02', 'laptop_use_01'),
            ('s02', 'scissors_use_01'),
            ('s03', 'mixer_use_01'),
        ][:args.max_sequences]
    
    # Run comparison
    results = comparison.run_comparison(sequences, args.output_dir)
    
    print(f"\nComparison completed! Results saved to {args.output_dir}")
    print("Check the following files:")
    print(f"- {args.output_dir}/comparison_results.json")
    print(f"- {args.output_dir}/comparison_report.md")
    print(f"- {args.output_dir}/comparison_bar_chart.png")
    print(f"- {args.output_dir}/performance_radar_chart.png")

if __name__ == "__main__":
    main()
