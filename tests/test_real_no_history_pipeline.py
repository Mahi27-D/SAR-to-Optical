import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SARToOpticalDataset
from src.models.generator import NoHistoryGenerator

def test_real_pipeline():
    index_path = os.path.join("outputs", "pilot_indexes_n3", "test_index.json")
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found.")
        sys.exit(1)
        
    with open(index_path, 'r') as f:
        sequences = json.load(f)
        
    dataset = SARToOpticalDataset(sequences)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    sar = batch['sar'].to(device)
    target_opt = batch['target_optical'].to(device)
    
    print(f"Input SAR shape: {sar.shape}")
    print(f"Target Optical shape: {target_opt.shape}")
    
    model = NoHistoryGenerator().to(device)
    
    # Forward pass
    output = model(sar)
    
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 4, 256, 256), f"Expected (2, 4, 256, 256), got {output.shape}"
    assert torch.isfinite(output).all(), "Output tensor contains non-finite values"
    
    # Loss
    criterion = nn.L1Loss()
    loss = criterion(output, target_opt)
    
    print(f"L1 Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradients found in {name}"
            
    print("PASS: Real data model smoke test passed. Forward and backward passes successful.")

if __name__ == "__main__":
    test_real_pipeline()
