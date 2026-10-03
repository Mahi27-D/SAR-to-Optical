import os
import tempfile
import torch
from torch.utils.data import DataLoader
import tifffile
import numpy as np
from datetime import datetime

from src.data.dataset import SARToOpticalDataset

def test_dataset_interface():
    print("=== Software/Interface Test for SARToOpticalDataset ===")
    print("Note: This is a synthetic test for the PyTorch interface ONLY, NOT real-data validation.\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary synthetic TIFFs mimicking real-world shapes
        s1_path = os.path.join(temp_dir, 'synthetic_s1.tif')
        s2_path = os.path.join(temp_dir, 'synthetic_s2.tif')
        
        # S1: 2 x 256 x 256
        s1_data = np.random.rand(2, 256, 256).astype(np.float32)
        tifffile.imwrite(s1_path, s1_data)
        
        # S2: 13 x 256 x 256
        s2_data = np.random.rand(13, 256, 256).astype(np.float32)
        tifffile.imwrite(s2_path, s2_data)
        
        # Construct synthetic causal sequence records
        # Emulating what extract_causal_sequences produces
        records = []
        for i in range(4): # 4 samples
            records.append({
                "target_index": i,
                "target_s2_timestamp": "2018-05-15",
                "target_s1_timestamp": "2018-05-15",
                "target_s1_path": s1_path,
                "target_s2_path": s2_path,
                "history": [
                    {
                        "history_index": 0,
                        "days_since": 30,
                        "month": 4,
                        "s2_timestamp": "2018-04-15",
                        "s2_path": "fake/history/path.tif" # The dataset doesn't load history pixels yet
                    }
                ]
            })
            
        dataset = SARToOpticalDataset(records)
        
        # Test 1: Single sample retrieval and shapes
        sample = dataset[0]
        assert "sar" in sample
        assert "target_optical" in sample
        assert "target_s1_timestamp" in sample
        assert "target_s2_timestamp" in sample
        assert "target_index" in sample
        assert "history" in sample
        
        print(f"Sample SAR shape: {sample['sar'].shape}")
        print(f"Sample Optical shape: {sample['target_optical'].shape}")
        
        assert sample['sar'].shape == (2, 256, 256), f"Expected (2, 256, 256), got {sample['sar'].shape}"
        assert sample['target_optical'].shape == (4, 256, 256), f"Expected (4, 256, 256), got {sample['target_optical'].shape}"
        
        # Test 2: Temporal metadata preservation
        target_time = datetime.strptime(sample['target_s2_timestamp'], "%Y-%m-%d")
        history = sample['history']
        assert len(history) == 1
        hist_time = datetime.strptime(history[0]['s2_timestamp'], "%Y-%m-%d")
        
        print(f"Target timestamp: {target_time.strftime('%Y-%m-%d')}")
        print(f"History timestamp: {hist_time.strftime('%Y-%m-%d')}")
        
        # Test 3: Historical timestamp is strictly earlier than target timestamp
        assert hist_time < target_time, "History timestamp must be strictly before target timestamp"
        
        # Test 4: DataLoader batching
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        print(f"Batch SAR shape: {batch['sar'].shape}")
        print(f"Batch Optical shape: {batch['target_optical'].shape}")
        
        assert batch['sar'].shape == (2, 2, 256, 256), f"Expected (2, 2, 256, 256), got {batch['sar'].shape}"
        assert batch['target_optical'].shape == (2, 4, 256, 256), f"Expected (2, 4, 256, 256), got {batch['target_optical'].shape}"
        
        print("\nAll dataset interface tests passed successfully!")

if __name__ == "__main__":
    test_dataset_interface()
