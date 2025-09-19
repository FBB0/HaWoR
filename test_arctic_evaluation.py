#!/usr/bin/env python3
"""
Test Script for ARCTIC Evaluation Framework
Simple test to verify the evaluation framework works
"""

import os
import sys
from pathlib import Path
import argparse

# Add HaWoR to path
sys.path.append(str(Path(__file__).parent))

from arctic_evaluation_framework import ArcticEvaluator, ArcticLossFunction
from arctic_comparison_script import ArcticBaselineComparison
from hawor_interface import HaWoRInterface

def test_loss_function():
    """Test the ARCTIC loss function"""
    print("🧪 Testing ARCTIC Loss Function...")
    
    loss_fn = ArcticLossFunction()
    
    # Create dummy data
    import torch
    
    # Dummy predictions
    pred_output = {
        'pred_keypoints_3d': torch.randn(2, 21, 3),
        'pred_keypoints_2d': torch.randn(2, 21, 2),
        'pred_mano_params': {
            'global_orient': torch.randn(2, 1, 3, 3),
            'hand_pose': torch.randn(2, 15, 3, 3),
            'betas': torch.randn(2, 10)
        }
    }
    
    # Dummy ground truth
    gt_data = {
        'gt_keypoints_3d': torch.randn(2, 21, 3),
        'gt_keypoints_2d': torch.randn(2, 21, 2),
        'gt_mano_params': {
            'global_orient': torch.randn(2, 1, 3, 3),
            'hand_pose': torch.randn(2, 15, 3, 3),
            'betas': torch.randn(2, 10)
        }
    }
    
    # Compute loss
    total_loss, losses = loss_fn.compute_total_loss(pred_output, gt_data)
    
    print(f"✅ Total Loss: {total_loss.item():.4f}")
    print("✅ Individual Losses:")
    for name, value in losses.items():
        print(f"   - {name}: {value.item():.4f}")
    
    return True

def test_hawor_interface():
    """Test HaWoR interface initialization"""
    print("\n🧪 Testing HaWoR Interface...")
    
    try:
        hawor_interface = HaWoRInterface(device='auto')
        hawor_interface.initialize_pipeline()
        print("✅ HaWoR Interface initialized successfully")
        return True
    except Exception as e:
        print(f"❌ HaWoR Interface failed: {e}")
        return False

def test_arctic_data_loading():
    """Test ARCTIC data loading"""
    print("\n🧪 Testing ARCTIC Data Loading...")
    
    arctic_root = "./thirdparty/arctic/unpack/arctic_data/data"
    
    if not Path(arctic_root).exists():
        print(f"❌ ARCTIC data not found at {arctic_root}")
        print("   Please ensure ARCTIC data is downloaded and extracted")
        return False
    
    try:
        hawor_interface = HaWoRInterface(device='auto')
        hawor_interface.initialize_pipeline()
        
        evaluator = ArcticEvaluator(hawor_interface, arctic_root)
        
        # Test loading a sequence
        arctic_data = evaluator.load_arctic_sequence('s01', 'box_grab_01')
        print("✅ ARCTIC sequence loaded successfully")
        print(f"   - MANO data keys: {list(arctic_data['mano'].keys())}")
        print(f"   - Egocam data keys: {list(arctic_data['egocam'].keys())}")
        
        # Test conversion to HaWoR format
        gt_data = evaluator.convert_arctic_to_hawor_format(arctic_data)
        print("✅ ARCTIC data converted to HaWoR format")
        print(f"   - GT data keys: {list(gt_data.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ ARCTIC data loading failed: {e}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics computation"""
    print("\n🧪 Testing Evaluation Metrics...")
    
    from arctic_evaluation_framework import ArcticEvaluationMetrics
    
    # Create dummy metrics
    metrics = ArcticEvaluationMetrics()
    metrics.mpjpe_3d = 5.2
    metrics.pck_3d_15mm = 0.85
    metrics.mpjpe_2d = 8.1
    metrics.pck_2d_10px = 0.78
    metrics.hand_detection_rate = 0.95
    
    # Convert to dict
    metrics_dict = metrics.to_dict()
    
    print("✅ Evaluation metrics created:")
    for key, value in metrics_dict.items():
        print(f"   - {key}: {value}")
    
    return True

def test_comparison_framework():
    """Test the comparison framework"""
    print("\n🧪 Testing Comparison Framework...")
    
    try:
        comparison = ArcticBaselineComparison()
        print("✅ Comparison framework initialized")
        print(f"   - ARCTIC baselines: {list(comparison.arctic_baselines.keys())}")
        
        # Test analysis function
        dummy_hawor_results = {
            'mpjpe_3d': {'mean': 7.5, 'std': 1.2},
            'pck_3d_15mm': {'mean': 0.88, 'std': 0.05},
            'mpjpe_2d': {'mean': 10.2, 'std': 2.1},
            'pck_2d_10px': {'mean': 0.82, 'std': 0.08}
        }
        
        analysis = comparison.analyze_comparison(dummy_hawor_results)
        print("✅ Comparison analysis completed")
        print(f"   - Strengths: {len(analysis['strengths'])}")
        print(f"   - Improvement areas: {len(analysis['improvement_areas'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison framework failed: {e}")
        return False

def run_quick_evaluation():
    """Run a quick evaluation test"""
    print("\n🚀 Running Quick Evaluation Test...")
    
    arctic_root = "./thirdparty/arctic/unpack/arctic_data/data"
    
    if not Path(arctic_root).exists():
        print(f"❌ ARCTIC data not found at {arctic_root}")
        return False
    
    try:
        # Initialize comparison framework
        comparison = ArcticBaselineComparison(arctic_root, device='auto')
        
        # Test with a single sequence
        sequences = [('s01', 'box_grab_01')]
        
        print(f"Evaluating sequence: {sequences[0]}")
        results = comparison.run_comparison(sequences, "./test_results")
        
        print("✅ Quick evaluation completed!")
        print("Results:")
        for metric, stats in results['hawor_results'].items():
            print(f"   - {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick evaluation failed: {e}")
        return False

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test ARCTIC Evaluation Framework')
    parser.add_argument('--quick-eval', action='store_true',
                       help='Run quick evaluation test')
    parser.add_argument('--all-tests', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    print("🧪 ARCTIC Evaluation Framework Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Loss Function
    total_tests += 1
    if test_loss_function():
        tests_passed += 1
    
    # Test 2: HaWoR Interface
    total_tests += 1
    if test_hawor_interface():
        tests_passed += 1
    
    # Test 3: Evaluation Metrics
    total_tests += 1
    if test_evaluation_metrics():
        tests_passed += 1
    
    # Test 4: Comparison Framework
    total_tests += 1
    if test_comparison_framework():
        tests_passed += 1
    
    # Test 5: ARCTIC Data Loading (optional)
    if args.all_tests:
        total_tests += 1
        if test_arctic_data_loading():
            tests_passed += 1
    
    # Test 6: Quick Evaluation (optional)
    if args.quick_eval:
        total_tests += 1
        if run_quick_evaluation():
            tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The evaluation framework is ready to use.")
        print("\nNext steps:")
        print("1. Ensure ARCTIC data is downloaded and extracted")
        print("2. Run: python arctic_evaluation_framework.py")
        print("3. Run: python arctic_comparison_script.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()
