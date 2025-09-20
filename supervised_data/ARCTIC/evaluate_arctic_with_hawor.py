#!/usr/bin/env python3
"""
Evaluate HaWoR on ARCTIC dataset
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import cv2
from tqdm import tqdm

# Add HaWoR to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from advanced_hawor import AdvancedHaWoR

class ArcticHaWoREvaluator:
    """Evaluate HaWoR on ARCTIC dataset"""
    
    def __init__(self, arctic_data_path: str, device: str = 'auto'):
        self.arctic_data_path = Path(arctic_data_path)
        self.device = device
        self.pipeline = AdvancedHaWoR(device=device)
        self.results = {}
        
        print(f"🔧 ARCTIC-HaWoR Evaluator initialized")
        print(f"📁 ARCTIC data path: {self.arctic_data_path}")
        print(f"🔧 Device: {device}")
    
    def find_arctic_sequences(self) -> List[Path]:
        """Find ARCTIC sequences to evaluate"""
        sequences = []
        
        # Look for processed sequences
        processed_dir = self.arctic_data_path / "arctic_data" / "data" / "splits"
        if processed_dir.exists():
            for split_dir in processed_dir.iterdir():
                if split_dir.is_dir():
                    for seq_file in split_dir.glob("*.npy"):
                        sequences.append(seq_file)
        
        # Look for raw sequences
        raw_dir = self.arctic_data_path / "arctic_data" / "data" / "raw_seqs"
        if raw_dir.exists():
            for subject_dir in raw_dir.iterdir():
                if subject_dir.is_dir():
                    for seq_file in subject_dir.glob("*.mano.npy"):
                        sequences.append(seq_file)
        
        return sequences
    
    def evaluate_sequence(self, seq_path: Path) -> Dict:
        """Evaluate a single ARCTIC sequence"""
        print(f"🎬 Evaluating sequence: {seq_path.name}")
        
        try:
            # Load ARCTIC sequence data
            seq_data = np.load(seq_path, allow_pickle=True).item()
            
            # Extract relevant information
            analysis = {
                'sequence_name': seq_path.name,
                'sequence_path': str(seq_path),
                'arctic_data': {
                    'has_mano_params': 'mano' in seq_data,
                    'has_smplx_params': 'smplx' in seq_data,
                    'has_object_params': 'object' in seq_data,
                    'has_camera_params': 'cam' in seq_data,
                    'frame_count': len(seq_data.get('mano', {}).get('right', {}).get('global_orient', [])) if 'mano' in seq_data else 0
                }
            }
            
            # TODO: Add HaWoR evaluation logic here
            # This would involve:
            # 1. Converting ARCTIC data to HaWoR format
            # 2. Running HaWoR inference
            # 3. Comparing with ARCTIC ground truth
            # 4. Computing evaluation metrics
            
            return analysis
            
        except Exception as e:
            return {
                'sequence_name': seq_path.name,
                'sequence_path': str(seq_path),
                'error': str(e)
            }
    
    def run_evaluation(self):
        """Run evaluation on all ARCTIC sequences"""
        print("🚀 Starting ARCTIC evaluation...")
        
        sequences = self.find_arctic_sequences()
        if not sequences:
            print("⚠️  No ARCTIC sequences found")
            return {}
        
        print(f"📊 Found {len(sequences)} sequences to evaluate")
        
        results = {}
        for seq_path in tqdm(sequences, desc="Evaluating sequences"):
            result = self.evaluate_sequence(seq_path)
            results[seq_path.name] = result
        
        # Save results
        results_file = self.supervised_data_root / "arctic_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved to: {results_file}")
        return results

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate HaWoR on ARCTIC dataset')
    parser.add_argument('--arctic-data', type=str, default='./thirdparty/arctic/data',
                       help='Path to ARCTIC data directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/mps/cpu/auto)')
    
    args = parser.parse_args()
    
    evaluator = ArcticHaWoREvaluator(args.arctic_data, args.device)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
