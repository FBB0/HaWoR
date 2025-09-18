# MANO Model Setup Instructions

## Overview

MANO (hand Model with Articulated and Non-rigid defOrmations) is required for HaWoR to generate hand meshes. Due to licensing requirements, MANO models must be downloaded manually from the official website.

## Step 1: Register and Download MANO Models

1. **Visit the MANO website**: https://mano.is.tue.mpg.de/
2. **Create an account** by clicking "Sign Up"
3. **Accept the license agreement**
4. **Download the models**:
   - Download `mano_v1_2.zip` (main MANO models)
   - The download includes:
     - `MANO_RIGHT.pkl` - Right hand model
     - `MANO_LEFT.pkl` - Left hand model

## Step 2: Extract and Place Files

After downloading, extract and place the files in the correct directory structure:

```bash
# Navigate to HaWoR directory
cd hawor-project/HaWoR

# Create MANO directories if they don't exist
mkdir -p _DATA/data/mano
mkdir -p _DATA/data_left/mano_left

# Extract the downloaded zip file and copy files:
# Right hand model
cp path/to/extracted/mano/models/MANO_RIGHT.pkl _DATA/data/mano/

# Left hand model
cp path/to/extracted/mano/models/MANO_LEFT.pkl _DATA/data_left/mano_left/
```

## Step 3: Verify Installation

Run this command to verify MANO models are correctly installed:

```bash
cd hawor-project/HaWoR
source ../../hawor/bin/activate
python -c "
import os
import sys
sys.path.insert(0, '.')

# Check file existence
right_path = '_DATA/data/mano/MANO_RIGHT.pkl'
left_path = '_DATA/data_left/mano_left/MANO_LEFT.pkl'

print('MANO Model Check:')
print(f'✅ Right hand model: {\"FOUND\" if os.path.exists(right_path) else \"MISSING\"}')
print(f'✅ Left hand model: {\"FOUND\" if os.path.exists(left_path) else \"MISSING\"}')

if os.path.exists(right_path) and os.path.exists(left_path):
    try:
        from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
        print('✅ MANO utilities can be imported')

        # Test basic MANO functionality
        import torch
        import numpy as np

        # Create dummy hand pose parameters
        trans = torch.zeros(1, 1, 3)
        rot = torch.zeros(1, 1, 3)
        hand_pose = torch.zeros(1, 1, 45)
        betas = torch.zeros(1, 1, 10)

        result = run_mano(trans, rot, hand_pose, betas)
        print(f'✅ MANO right hand generation: SUCCESS (vertices shape: {result[\"vertices\"].shape})')

        result_left = run_mano_left(trans, rot, hand_pose, betas)
        print(f'✅ MANO left hand generation: SUCCESS (vertices shape: {result_left[\"vertices\"].shape})')

        print()
        print('🎉 MANO setup is complete and functional!')

    except Exception as e:
        print(f'❌ MANO functionality test failed: {e}')
        print('Please check that the MANO model files are valid.')
else:
    print()
    print('❌ MANO models not found. Please follow the setup instructions above.')
"
```

## Expected Directory Structure

After setup, your directory should look like this:

```
hawor-project/HaWoR/
├── _DATA/
│   ├── data/
│   │   └── mano/
│   │       └── MANO_RIGHT.pkl
│   └── data_left/
│       └── mano_left/
│           └── MANO_LEFT.pkl
└── ... (other HaWoR files)
```

## Troubleshooting

### Common Issues:

1. **"No such file or directory" errors**:
   - Ensure the directory structure matches exactly
   - Check file paths are correct (case-sensitive)

2. **"Invalid pickle file" errors**:
   - Re-download the MANO models
   - Ensure files weren't corrupted during download/transfer

3. **Import errors**:
   - Make sure you're in the correct virtual environment
   - Verify all dependencies are installed

### Testing Without MANO Models

If you don't have MANO models yet, you can still test HaWoR with our simplified pipeline:

```bash
python hawor_interface.py --video example/video_0.mp4 --mode simplified
```

## License Information

MANO models are subject to their own license terms. By downloading and using MANO models, you agree to abide by the [MANO license](https://mano.is.tue.mpg.de/license.html).

## Next Steps

Once MANO models are installed:

1. Test the full HaWoR pipeline:
   ```bash
   python hawor_interface.py --video example/video_0.mp4 --mode hawor
   ```

2. Process your own videos:
   ```bash
   python hawor_interface.py --video /path/to/your/video.mp4
   ```

3. Use batch processing:
   ```bash
   python hawor_interface.py --videos video1.mp4 video2.mp4 video3.mp4
   ```