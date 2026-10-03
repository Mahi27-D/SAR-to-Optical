import os
import tempfile
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import tifffile
from datetime import datetime

from src.training.train_no_history import load_index
from src.data.dataset import SARToOpticalDataset

def test_contract():
    print("=== Index -> Dataset contract test ===")
    print("Synthetic software validation")
    print("NOT real-data validation")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Create temporary synthetic TIFF files
        s1_path = os.path.join(temp_dir, 'synthetic_s1.tif')
        s2_path = os.path.join(temp_dir, 'synthetic_s2.tif')
        
        # S1: 2 x 256 x 256
        s1_data = np.random.rand(2, 256, 256).astype(np.float32)
        tifffile.imwrite(s1_path, s1_data)
        
        # S2: 13 x 256 x 256
        s2_data = np.random.rand(13, 256, 256).astype(np.float32)
        tifffile.imwrite(s2_path, s2_data)
        
        # 2. Construct synthetic causal-index record using the EXACT schema produced by temporal_index.py
        # temporal_index.py valid_sequences.append(...) output schema:
        # {
        #   "target_index": int,
        #   "target_s2_timestamp": str,
        #   "target_s1_timestamp": str,
        #   "target_s2_path": str,
        #   "target_s1_path": str,
        #   "history": [
        #       {
        #           "history_index": int,
        #           "days_since": int,
        #           "month": int,
        #           "s2_timestamp": str
        #       }
        #   ]
        # }
        
        records = []
        for i in range(2):
            records.append({
                "target_index": 20,
                "target_s2_timestamp": "2018-05-15",
                "target_s1_timestamp": "2018-05-15",
                "target_s2_path": s2_path,
                "target_s1_path": s1_path,
                "history": [
                    {
                        "history_index": 15,
                        "days_since": 30,
                        "month": 4,
                        "s2_timestamp": "2018-04-15"
                    }
                ]
            })
            
        # 3. Write record to temporary JSON file
        index_json_path = os.path.join(temp_dir, 'train_index.json')
        with open(index_json_path, 'w') as f:
            json.dump(records, f)
            
        # 4. Load JSON using load_index() from train_no_history.py
        loaded_records = load_index(index_json_path)
        
        # 5. Pass into SARToOpticalDataset
        dataset = SARToOpticalDataset(loaded_records)
        
        # 6. Verify Dataset[0] returns required keys
        sample = dataset[0]
        expected_keys = ["sar", "target_optical", "target_s1_timestamp", 
                         "target_s2_timestamp", "target_index", "history"]
                         
        for key in expected_keys:
            assert key in sample, f"Contract mismatch: Missing key '{key}' in dataset output"
            
        # 7. Verify tensor shapes
        assert sample["sar"].shape == (2, 256, 256), f"Mismatch SAR shape: {sample['sar'].shape}"
        assert sample["target_optical"].shape == (4, 256, 256), f"Mismatch Optical shape: {sample['target_optical'].shape}"
        
        # 8. Verify history timestamp strictly earlier than target timestamp
        target_ts = datetime.strptime(sample["target_s2_timestamp"], "%Y-%m-%d")
        for h in sample["history"]:
            hist_ts = datetime.strptime(h["s2_timestamp"], "%Y-%m-%d")
            assert hist_ts < target_ts, f"Causal violation in metadata: History {hist_ts} >= Target {target_ts}"
            
        # 9. Create DataLoader with batch_size=2
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        # 10. Verify batched tensor shapes
        assert batch["sar"].shape == (2, 2, 256, 256), f"Mismatch Batch SAR shape: {batch['sar'].shape}"
        assert batch["target_optical"].shape == (2, 4, 256, 256), f"Mismatch Batch Optical shape: {batch['target_optical'].shape}"
        
        print("Dataset successfully ingested the temporal_index JSON schema.")
        print("Tensor shapes and causal invariants verified.")
        print("Contract Test Passed.")

if __name__ == "__main__":
    test_contract()
