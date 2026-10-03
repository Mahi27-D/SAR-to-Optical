import os
import sys
import tempfile
import json
import numpy as np
import tifffile
import torch
from unittest.mock import patch

from src.training.train_no_history import main
from src.models.generator import NoHistoryGenerator

def test_training_smoke():
    print("Starting training infrastructure smoke test...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create synthetic TIFFs
        s1_path = os.path.join(temp_dir, 'synthetic_s1.tif')
        s2_path = os.path.join(temp_dir, 'synthetic_s2.tif')
        
        s1_data = np.random.rand(2, 256, 256).astype(np.float32)
        s2_data = np.random.rand(13, 256, 256).astype(np.float32)
        
        tifffile.imwrite(s1_path, s1_data)
        tifffile.imwrite(s2_path, s2_data)
        
        # Create index records
        record = {
            "target_index": 1,
            "target_s2_timestamp": "2018-05-15",
            "target_s1_timestamp": "2018-05-15",
            "target_s2_path": s2_path,
            "target_s1_path": s1_path,
            "history": []
        }
        records = [record, record] # Batch size 2 needs at least 2 records
        
        train_index_path = os.path.join(temp_dir, 'train_index.json')
        val_index_path = os.path.join(temp_dir, 'val_index.json')
        
        with open(train_index_path, 'w') as f:
            json.dump(records, f)
        with open(val_index_path, 'w') as f:
            json.dump(records, f)
            
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        
        test_args = [
            'train_no_history.py',
            '--train_index', train_index_path,
            '--val_index', val_index_path,
            '--data_root', temp_dir,
            '--epochs', '2',
            '--batch_size', '2',
            '--checkpoint_dir', checkpoint_dir
        ]
        
        print("Running train_no_history.py main()...")
        with patch.object(sys, 'argv', test_args):
            main()
            
        print("Verifying checkpoint creation...")
        assert os.path.exists(os.path.join(checkpoint_dir, 'latest.pt')), "latest.pt not found"
        assert os.path.exists(os.path.join(checkpoint_dir, 'best.pt')), "best.pt not found"
        
        print("Loading best.pt checkpoint...")
        model = NoHistoryGenerator()
        # weights_only=True may be safer for torch.load, but weights_only=False ensures backward compatibility if state_dict has non-tensors
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best.pt'), weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Testing forward pass on loaded model...")
        model.eval()
        dummy_input = torch.randn(2, 2, 256, 256)
        with torch.no_grad():
            out = model(dummy_input)
            
        assert out.shape == (2, 4, 256, 256), f"Expected (2, 4, 256, 256), got {out.shape}"
        print("PASS: Training infrastructure smoke test passed successfully.")

if __name__ == "__main__":
    test_training_smoke()
