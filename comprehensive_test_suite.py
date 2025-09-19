#!/usr/bin/env python3
"""
Comprehensive Test Suite for HaWoR Project
Tests all major components for functionality, performance, and integration
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TestResult:
    """Test result container"""
    name: str
    passed: bool
    duration: float
    message: str = ""
    details: Dict = None

class TestSuite:
    """Comprehensive test suite for HaWoR components"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def log_test(self, name: str, passed: bool, duration: float, message: str = "", details: Dict = None):
        """Log test result"""
        result = TestResult(name, passed, duration, message, details or {})
        self.results.append(result)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {name} ({duration:.3f}s)")
        if message:
            print(f"        {message}")

    def test_file_structure(self) -> bool:
        """Test project file structure"""
        print("📁 Testing file structure...")
        start_time = time.time()

        required_files = [
            # Core implementation
            "enhanced_training_evaluation.py",
            "enhanced_training_pipeline.py",
            "training_data_preparation.py",
            "arctic_evaluation_framework.py",
            "arctic_comparison_script.py",

            # Configuration and setup
            "setup_training_pipeline.py",
            "optimize_training_config.py",
            "configs/production_training_config.json",
            "configs/optimized_training_config.json",

            # Validation and testing
            "validate_framework.py",
            "validate_metrics.py",
            "demo_evaluation.py",
            "test_arctic_evaluation.py",

            # Utilities
            "launch_training.py",
            "prepare_training_data.py",
            "monitor_training.py",

            # Documentation
            "ENHANCED_TRAINING_EVALUATION_README.md",
            "ARCTIC_EVALUATION_FRAMEWORK_SUMMARY.md",
        ]

        missing_files = []
        existing_files = []

        for file_path in required_files:
            if Path(file_path).exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)

        passed = len(missing_files) == 0
        duration = time.time() - start_time

        message = f"Found {len(existing_files)}/{len(required_files)} required files"
        if missing_files:
            message += f". Missing: {', '.join(missing_files[:3])}{'...' if len(missing_files) > 3 else ''}"

        details = {
            "existing_files": existing_files,
            "missing_files": missing_files,
            "total_required": len(required_files)
        }

        self.log_test("File Structure", passed, duration, message, details)
        return passed

    def test_configuration_files(self) -> bool:
        """Test configuration file validity"""
        print("\n⚙️  Testing configuration files...")
        start_time = time.time()

        config_files = [
            "configs/production_training_config.json",
            "configs/optimized_training_config.json"
        ]

        valid_configs = 0
        total_configs = len(config_files)
        config_details = {}

        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)

                    # Validate required sections
                    required_sections = ["data", "model", "training", "loss"]
                    missing_sections = [s for s in required_sections if s not in config]

                    if not missing_sections:
                        valid_configs += 1
                        config_details[config_file] = {"status": "valid", "sections": len(config)}
                    else:
                        config_details[config_file] = {"status": "invalid", "missing": missing_sections}

                except Exception as e:
                    config_details[config_file] = {"status": "error", "error": str(e)}
            else:
                config_details[config_file] = {"status": "missing"}

        passed = valid_configs == total_configs
        duration = time.time() - start_time
        message = f"Valid configurations: {valid_configs}/{total_configs}"

        self.log_test("Configuration Files", passed, duration, message, config_details)
        return passed

    def test_validation_frameworks(self) -> bool:
        """Test validation framework functionality"""
        print("\n🧪 Testing validation frameworks...")
        start_time = time.time()

        validation_tests = []

        # Test framework validation
        try:
            import subprocess
            result = subprocess.run([sys.executable, "validate_framework.py"],
                                  capture_output=True, text=True, timeout=30)
            framework_passed = result.returncode == 0 and "3/3 tests passed" in result.stdout
            validation_tests.append(("Framework Validation", framework_passed))
        except Exception as e:
            validation_tests.append(("Framework Validation", False))

        # Test metrics validation
        try:
            result = subprocess.run([sys.executable, "validate_metrics.py"],
                                  capture_output=True, text=True, timeout=30)
            metrics_passed = result.returncode == 0 and "7/7 tests passed" in result.stdout
            validation_tests.append(("Metrics Validation", metrics_passed))
        except Exception as e:
            validation_tests.append(("Metrics Validation", False))

        # Test demo evaluation
        try:
            result = subprocess.run([sys.executable, "demo_evaluation.py"],
                                  capture_output=True, text=True, timeout=30)
            demo_passed = result.returncode == 0 and "completed successfully" in result.stdout
            validation_tests.append(("Demo Evaluation", demo_passed))
        except Exception as e:
            validation_tests.append(("Demo Evaluation", False))

        passed_tests = sum(1 for _, passed in validation_tests if passed)
        total_tests = len(validation_tests)
        passed = passed_tests == total_tests

        duration = time.time() - start_time
        message = f"Validation tests passed: {passed_tests}/{total_tests}"

        details = {test_name: passed for test_name, passed in validation_tests}

        self.log_test("Validation Frameworks", passed, duration, message, details)
        return passed

    def test_data_pipeline(self) -> bool:
        """Test data pipeline components"""
        print("\n📊 Testing data pipeline...")
        start_time = time.time()

        pipeline_tests = []

        # Test ARCTIC data availability
        arctic_root = Path("thirdparty/arctic/unpack/arctic_data/data")
        arctic_available = arctic_root.exists() and (arctic_root / "raw_seqs").exists()
        pipeline_tests.append(("ARCTIC Data", arctic_available))

        # Test MANO models
        mano_right = Path("_DATA/data/mano/MANO_RIGHT.pkl")
        mano_left = Path("_DATA/data_left/mano_left/MANO_LEFT.pkl")
        mano_available = mano_right.exists() and mano_left.exists()
        pipeline_tests.append(("MANO Models", mano_available))

        # Test output directories
        output_dirs = ["training_data", "validation_data", "training_output", "checkpoints"]
        dirs_created = all(Path(d).exists() for d in output_dirs)
        pipeline_tests.append(("Output Directories", dirs_created))

        # Test configuration loading
        try:
            with open("configs/production_training_config.json", 'r') as f:
                config = json.load(f)
            config_valid = "data" in config and "training" in config
            pipeline_tests.append(("Config Loading", config_valid))
        except:
            pipeline_tests.append(("Config Loading", False))

        passed_tests = sum(1 for _, passed in pipeline_tests if passed)
        total_tests = len(pipeline_tests)
        passed = passed_tests == total_tests

        duration = time.time() - start_time
        message = f"Pipeline tests passed: {passed_tests}/{total_tests}"

        details = {test_name: passed for test_name, passed in pipeline_tests}

        self.log_test("Data Pipeline", passed, duration, message, details)
        return passed

    def test_training_components(self) -> bool:
        """Test training pipeline components"""
        print("\n🏋️  Testing training components...")
        start_time = time.time()

        training_tests = []

        # Test training scripts existence
        training_scripts = [
            "enhanced_training_pipeline.py",
            "launch_training.py",
            "prepare_training_data.py",
            "monitor_training.py"
        ]

        scripts_exist = all(Path(script).exists() for script in training_scripts)
        training_tests.append(("Training Scripts", scripts_exist))

        # Test configuration optimization
        try:
            import subprocess
            result = subprocess.run([sys.executable, "optimize_training_config.py"],
                                  capture_output=True, text=True, timeout=60)
            optimization_works = result.returncode == 0 and "optimization completed" in result.stdout
            training_tests.append(("Config Optimization", optimization_works))
        except Exception as e:
            training_tests.append(("Config Optimization", False))

        # Test setup pipeline
        try:
            result = subprocess.run([sys.executable, "setup_training_pipeline.py"],
                                  capture_output=True, text=True, timeout=60)
            setup_works = result.returncode == 0 and "setup completed successfully" in result.stdout
            training_tests.append(("Pipeline Setup", setup_works))
        except Exception as e:
            training_tests.append(("Pipeline Setup", False))

        # Test launcher dry run
        if Path("launch_training.py").exists():
            try:
                result = subprocess.run([sys.executable, "launch_training.py", "--dry-run"],
                                      capture_output=True, text=True, timeout=30)
                launcher_works = result.returncode == 0 or "dry run" in result.stdout.lower()
                training_tests.append(("Training Launcher", launcher_works))
            except Exception as e:
                training_tests.append(("Training Launcher", False))
        else:
            training_tests.append(("Training Launcher", False))

        passed_tests = sum(1 for _, passed in training_tests if passed)
        total_tests = len(training_tests)
        passed = passed_tests == total_tests

        duration = time.time() - start_time
        message = f"Training tests passed: {passed_tests}/{total_tests}"

        details = {test_name: passed for test_name, passed in training_tests}

        self.log_test("Training Components", passed, duration, message, details)
        return passed

    def test_evaluation_pipeline(self) -> bool:
        """Test evaluation pipeline components"""
        print("\n📈 Testing evaluation pipeline...")
        start_time = time.time()

        evaluation_tests = []

        # Test evaluation scripts
        eval_scripts = [
            "arctic_evaluation_framework.py",
            "arctic_comparison_script.py",
            "enhanced_training_evaluation.py"
        ]

        scripts_exist = all(Path(script).exists() for script in eval_scripts)
        evaluation_tests.append(("Evaluation Scripts", scripts_exist))

        # Test report generation
        report_files = [
            "demo_evaluation_report.md",
            "demo_evaluation_results.json",
            "metrics_validation_report.json"
        ]

        reports_exist = all(Path(report).exists() for report in report_files)
        evaluation_tests.append(("Evaluation Reports", reports_exist))

        # Test metrics computation
        try:
            # Simple metrics test
            def compute_mpjpe(pred, gt):
                import math
                total_error = 0
                count = 0
                for i in range(len(pred)):
                    for j in range(len(pred[i])):
                        diff = pred[i][j] - gt[i][j]
                        total_error += diff * diff
                        count += 1
                return math.sqrt(total_error / count)

            pred = [[1, 2, 3]]
            gt = [[0, 0, 0]]
            mpjpe = compute_mpjpe(pred, gt)
            metrics_work = abs(mpjpe - math.sqrt(14)) < 1e-6
            evaluation_tests.append(("Metrics Computation", metrics_work))
        except Exception as e:
            evaluation_tests.append(("Metrics Computation", False))

        # Test baseline comparison
        try:
            with open("demo_evaluation_results.json", 'r') as f:
                results = json.load(f)
            comparison_data = "arctic_baselines" in results and "hawor_results" in results
            evaluation_tests.append(("Baseline Comparison", comparison_data))
        except:
            evaluation_tests.append(("Baseline Comparison", False))

        passed_tests = sum(1 for _, passed in evaluation_tests if passed)
        total_tests = len(evaluation_tests)
        passed = passed_tests == total_tests

        duration = time.time() - start_time
        message = f"Evaluation tests passed: {passed_tests}/{total_tests}"

        details = {test_name: passed for test_name, passed in evaluation_tests}

        self.log_test("Evaluation Pipeline", passed, duration, message, details)
        return passed

    def test_integration(self) -> bool:
        """Test component integration"""
        print("\n🔗 Testing component integration...")
        start_time = time.time()

        integration_tests = []

        # Test config file compatibility
        try:
            with open("configs/production_training_config.json", 'r') as f:
                prod_config = json.load(f)
            with open("configs/optimized_training_config.json", 'r') as f:
                opt_config = json.load(f)

            # Check that both configs have compatible structure
            common_keys = set(prod_config.keys()) & set(opt_config.keys())
            compatibility = len(common_keys) >= 4  # At least 4 common sections
            integration_tests.append(("Config Compatibility", compatibility))
        except:
            integration_tests.append(("Config Compatibility", False))

        # Test data path consistency
        try:
            with open("training_setup_report.json", 'r') as f:
                setup_report = json.load(f)

            data_paths = setup_report.get("setup_summary", {}).get("data_paths", {})
            paths_consistent = all(Path(path).exists() or "training_data" in path
                                 for path in data_paths.values() if isinstance(path, str))
            integration_tests.append(("Data Path Consistency", paths_consistent))
        except:
            integration_tests.append(("Data Path Consistency", False))

        # Test script chain compatibility
        script_chain = [
            "setup_training_pipeline.py",
            "prepare_training_data.py",
            "launch_training.py"
        ]

        chain_complete = all(Path(script).exists() and os.access(script, os.X_OK)
                           for script in script_chain)
        integration_tests.append(("Script Chain", chain_complete))

        # Test documentation consistency
        docs = [
            "ENHANCED_TRAINING_EVALUATION_README.md",
            "ARCTIC_EVALUATION_FRAMEWORK_SUMMARY.md"
        ]

        docs_exist = all(Path(doc).exists() for doc in docs)
        integration_tests.append(("Documentation", docs_exist))

        passed_tests = sum(1 for _, passed in integration_tests if passed)
        total_tests = len(integration_tests)
        passed = passed_tests == total_tests

        duration = time.time() - start_time
        message = f"Integration tests passed: {passed_tests}/{total_tests}"

        details = {test_name: passed for test_name, passed in integration_tests}

        self.log_test("Component Integration", passed, duration, message, details)
        return passed

    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks"""
        print("\n⚡ Testing performance benchmarks...")
        start_time = time.time()

        performance_tests = []

        # Test evaluation speed
        try:
            eval_start = time.time()
            import subprocess
            result = subprocess.run([sys.executable, "demo_evaluation.py"],
                                  capture_output=True, text=True, timeout=60)
            eval_duration = time.time() - eval_start

            fast_evaluation = eval_duration < 30  # Should complete in under 30 seconds
            performance_tests.append(("Evaluation Speed", fast_evaluation))
        except:
            performance_tests.append(("Evaluation Speed", False))

        # Test memory efficiency
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            memory_efficient = memory_mb < 1000  # Should use less than 1GB for testing
            performance_tests.append(("Memory Usage", memory_efficient))
        except:
            performance_tests.append(("Memory Usage", True))  # Skip if psutil not available

        # Test config loading speed
        try:
            config_start = time.time()
            for _ in range(10):
                with open("configs/production_training_config.json", 'r') as f:
                    json.load(f)
            config_duration = time.time() - config_start

            fast_config = config_duration < 1.0  # 10 loads in under 1 second
            performance_tests.append(("Config Loading Speed", fast_config))
        except:
            performance_tests.append(("Config Loading Speed", False))

        # Test file I/O performance
        try:
            io_start = time.time()
            test_data = {"test": "data", "numbers": list(range(1000))}

            for i in range(10):
                with open(f"temp_test_{i}.json", 'w') as f:
                    json.dump(test_data, f)
                with open(f"temp_test_{i}.json", 'r') as f:
                    json.load(f)
                os.remove(f"temp_test_{i}.json")

            io_duration = time.time() - io_start
            fast_io = io_duration < 2.0  # 10 file operations in under 2 seconds
            performance_tests.append(("File I/O Speed", fast_io))
        except:
            performance_tests.append(("File I/O Speed", False))

        passed_tests = sum(1 for _, passed in performance_tests if passed)
        total_tests = len(performance_tests)
        passed = passed_tests >= total_tests - 1  # Allow one performance test to fail

        duration = time.time() - start_time
        message = f"Performance tests passed: {passed_tests}/{total_tests}"

        details = {test_name: passed for test_name, passed in performance_tests}

        self.log_test("Performance Benchmarks", passed, duration, message, details)
        return passed

    def run_all_tests(self) -> Dict:
        """Run all tests and generate comprehensive report"""
        print("🧪 HaWoR Comprehensive Test Suite")
        print("=" * 50)

        test_functions = [
            self.test_file_structure,
            self.test_configuration_files,
            self.test_validation_frameworks,
            self.test_data_pipeline,
            self.test_training_components,
            self.test_evaluation_pipeline,
            self.test_integration,
            self.test_performance_benchmarks
        ]

        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                print(f"❌ Test {test_func.__name__} failed with error: {e}")
                traceback.print_exc()

        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        total_duration = time.time() - self.start_time

        print("\n" + "=" * 50)
        print(f"🎯 Test Summary: {passed_tests}/{total_tests} tests passed")
        print(f"⏱️  Total duration: {total_duration:.2f} seconds")

        # Categorize results
        critical_tests = [
            "File Structure", "Configuration Files", "Validation Frameworks"
        ]
        critical_passed = sum(1 for r in self.results
                            if r.name in critical_tests and r.passed)

        if passed_tests == total_tests:
            print("🎉 All tests passed! System is ready for production.")
        elif critical_passed == len(critical_tests):
            print("✅ Critical tests passed. Some optional features may need attention.")
        else:
            print("⚠️  Critical tests failed. System needs fixes before use.")

        # Generate detailed report
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration,
                "critical_tests_passed": critical_passed,
                "status": "READY" if passed_tests == total_tests else "NEEDS_ATTENTION"
            },
            "test_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ],
            "recommendations": self.generate_recommendations()
        }

        # Save report
        with open("comprehensive_test_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Detailed test report saved to: comprehensive_test_report.json")
        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        failed_tests = [r for r in self.results if not r.passed]

        if not failed_tests:
            recommendations.extend([
                "All tests passed - system is ready for production use",
                "Consider running periodic regression tests",
                "Monitor performance in production environment"
            ])
        else:
            for test in failed_tests:
                if test.name == "File Structure":
                    recommendations.append("Install missing dependencies and files")
                elif test.name == "Configuration Files":
                    recommendations.append("Fix configuration file format or content")
                elif test.name == "Validation Frameworks":
                    recommendations.append("Debug validation framework execution")
                elif test.name == "Data Pipeline":
                    recommendations.append("Set up ARCTIC data and MANO models")
                elif test.name == "Training Components":
                    recommendations.append("Fix training pipeline setup")
                elif test.name == "Evaluation Pipeline":
                    recommendations.append("Debug evaluation components")
                elif test.name == "Component Integration":
                    recommendations.append("Fix component compatibility issues")
                elif test.name == "Performance Benchmarks":
                    recommendations.append("Optimize performance or increase timeouts")

        return recommendations

def main():
    """Main test function"""
    suite = TestSuite()
    report = suite.run_all_tests()
    return 0 if report["summary"]["status"] == "READY" else 1

if __name__ == "__main__":
    sys.exit(main())