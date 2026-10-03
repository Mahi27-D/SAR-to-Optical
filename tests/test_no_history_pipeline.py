import os
import tempfile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import tifffile

from src.data.dataset import SARToOpticalDataset
from src.models.generator import NoHistoryGenerator

def test_no_history_pipeline():
    print("=== No-History end-to-end synthetic smoke test ===")
    print("NOT real-data validation")
    print("NOT a trained model result")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        
        # 2. Create at least 2 synthetic causal records
        records = []
        for i in range(2):
            records.append({
                "target_index": i,
                "target_s2_timestamp": "2018-05-15",
                "target_s1_timestamp": "2018-05-15",
                "target_s1_path": s1_path,
                "target_s2_path": s2_path,
                "history": []
            })
            
        # 3. Load them using the EXISTING SARToOpticalDataset
        dataset = SARToOpticalDataset(records)
        
        # 4. Use DataLoader(batch_size=2, shuffle=False)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        sar_batch = batch['sar'].to(device)
        target_optical_batch = batch['target_optical'].to(device)
        
        # 5. Instantiate the EXISTING No-History Generator
        generator = NoHistoryGenerator().to(device)
        
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
        
        # 7. Run forward pass
        generated = generator(sar_batch)
        
        # 8. Verify shape
        assert generated.shape == (2, 4, 256, 256), f"Expected shape (2, 4, 256, 256), got {generated.shape}"
        
        # 9. Compute temporary L1 reconstruction loss
        loss = F.l1_loss(generated, target_optical_batch)
        
        # 10. Run backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        
        gradients_exist_and_finite = True
        for param in generator.parameters():
            if param.requires_grad:
                if param.grad is None or not torch.isfinite(param.grad).all():
                    gradients_exist_and_finite = False
                    break
                    
        optimizer_step_successful = False
        try:
            optimizer.step()
            optimizer_step_successful = True
        except Exception as e:
            print(f"Optimizer step failed: {e}")
            
        # 11 & 12. Print explicitly requested info
        total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        
        print(f"Input SAR shape: {sar_batch.shape}")
        print(f"Target optical shape: {target_optical_batch.shape}")
        print(f"Generated optical shape: {generated.shape}")
        print(f"Parameter count: {total_params:,}")
        print(f"CUDA device: {device}")
        print(f"Loss: {loss.item():.6f}")
        print(f"Gradients finite: {gradients_exist_and_finite}")
        print(f"Optimizer step successful: {optimizer_step_successful}")

if __name__ == "__main__":
    test_no_history_pipeline()
