import os
import sys
import json
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SARToOpticalDataset

def test_real_index():
    index_path = os.path.join("outputs", "trainval_indexes_n3", "train_index.json")
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found.")
        sys.exit(1)
        
    with open(index_path, 'r') as f:
        sequences = json.load(f)
        
    print(f"Loaded {len(sequences)} sequences from {index_path}")
    
    # Instantiate dataset
    dataset = SARToOpticalDataset(sequences)
    
    # Load first sample
    sample = dataset[0]
    
    sar = sample['sar']
    opt = sample['target_optical']
    hist_opt = sample['history_optical']
    
    print(f"Single sample SAR shape: {sar.shape}")
    print(f"Single sample Optical shape: {opt.shape}")
    print(f"Single sample History Optical shape: {hist_opt.shape}")
    
    assert sar.shape == (2, 256, 256), f"Expected (2, 256, 256), got {sar.shape}"
    assert opt.shape == (4, 256, 256), f"Expected (4, 256, 256), got {opt.shape}"
    assert hist_opt.shape == (3, 4, 256, 256), f"Expected (3, 4, 256, 256), got {hist_opt.shape}"
    
    assert torch.isfinite(sar).all(), "SAR tensor contains non-finite values"
    assert torch.isfinite(opt).all(), "Optical tensor contains non-finite values"
    assert torch.isfinite(hist_opt).all(), "History optical tensor contains non-finite values"
    
    # DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    batch_sar = batch['sar']
    batch_opt = batch['target_optical']
    batch_hist_opt = batch['history_optical']
    
    print(f"Batch SAR shape: {batch_sar.shape}")
    print(f"Batch Optical shape: {batch_opt.shape}")
    print(f"Batch History Optical shape: {batch_hist_opt.shape}")
    
    assert batch_sar.shape == (2, 2, 256, 256), f"Expected (2, 2, 256, 256), got {batch_sar.shape}"
    assert batch_opt.shape == (2, 4, 256, 256), f"Expected (2, 4, 256, 256), got {batch_opt.shape}"
    assert batch_hist_opt.shape == (2, 3, 4, 256, 256), f"Expected (2, 3, 4, 256, 256), got {batch_hist_opt.shape}"
    
    print("PASS: Real index loaded successfully and tensors verified.")

if __name__ == "__main__":
    test_real_index()
