import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.generator import NoHistoryGenerator
from src.data.dataset import SARToOpticalDataset

def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file missing: {index_path}")
    with open(index_path, 'r') as f:
        return json.load(f)

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    
    total_l1_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    print("Starting evaluation...")
    for batch_idx, batch in enumerate(dataloader):
        sar = batch['sar'].to(device)
        target = batch['target_optical'].to(device)
        
        output = model(sar)
        loss = F.l1_loss(output, target)
        
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss encountered at batch {batch_idx}")
            continue
            
        B = output.shape[0]
        
        output_np = output.cpu().numpy()
        target_np = target.cpu().numpy()
        
        for i in range(B):
            pred_img = output_np[i]
            true_img = target_np[i]
            
            # Format: [C, H, W] -> [H, W, C]
            pred_img = np.transpose(pred_img, (1, 2, 0))
            true_img = np.transpose(true_img, (1, 2, 0))
            
            drange = 10000.0
                
            batch_psnr = psnr(true_img, pred_img, data_range=drange)
            batch_ssim = ssim(true_img, pred_img, data_range=drange, channel_axis=-1)
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_samples += 1
            
        total_l1_loss += loss.item() * B
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Evaluated {batch_idx + 1}/{len(dataloader)} batches. Samples done: {num_samples}")
            
    if num_samples == 0:
        raise ValueError("No valid samples evaluated.")
        
    avg_l1 = total_l1_loss / num_samples
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return {
        "mean_l1_loss": float(avg_l1),
        "psnr": float(avg_psnr),
        "ssim": float(avg_ssim),
        "num_evaluated_samples": int(num_samples)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate No-History Baseline")
    parser.add_argument('--test_index', type=str, default='outputs/pilot_indexes_n3/test_index.json', help="Path to test index.")
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/no_history_n3/best.pt', help="Path to best checkpoint.")
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation/', help="Directory to save results.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers.")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Selected: {device}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    test_records = load_index(args.test_index)
    
    test_dataset = SARToOpticalDataset(test_records)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = NoHistoryGenerator().to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    results = evaluate_model(model, test_loader, device)
    
    print("\n--- Evaluation Results ---")
    print(f"No-History Baseline test results:")
    print(f"Successfully evaluated samples: {results['num_evaluated_samples']}")
    print(f"Mean L1 Loss: {results['mean_l1_loss']:.4f}")
    print(f"PSNR: {results['psnr']:.4f}")
    print(f"SSIM: {results['ssim']:.4f}")
    print("--------------------------\n")
    
    out_path = os.path.join(args.output_dir, 'no_history_baseline_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
