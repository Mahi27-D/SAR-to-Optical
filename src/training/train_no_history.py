import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import shutil

from src.models.generator import NoHistoryGenerator
from src.data.dataset import SARToOpticalDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, is_best, checkpoint_dir, filename='latest.pt'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best.pt')
        shutil.copyfile(filepath, best_filepath)

def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file missing: {index_path}")
    with open(index_path, 'r') as f:
        return json.load(f)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    if len(dataloader) == 0:
        raise ValueError("Training dataloader is empty.")
        
    for batch_idx, batch in enumerate(dataloader):
        sar = batch['sar'].to(device)
        target = batch['target_optical'].to(device)
        
        optimizer.zero_grad()
        output = model(sar)
        loss = F.l1_loss(output, target)
        
        if not torch.isfinite(loss):
            raise ValueError(f"Loss is {loss.item()} (not finite). Stopping training.")
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

@torch.no_grad()
def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    
    if len(dataloader) == 0:
        raise ValueError("Validation dataloader is empty.")
        
    for batch_idx, batch in enumerate(dataloader):
        sar = batch['sar'].to(device)
        target = batch['target_optical'].to(device)
        
        output = model(sar)
        loss = F.l1_loss(output, target)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="No-History Baseline Training Infrastructure")
    parser.add_argument('--train_index', type=str, required=True, help="Path to JSON file containing causal sequence records for training.")
    parser.add_argument('--val_index', type=str, required=True, help="Path to JSON file containing causal sequence records for validation.")
    parser.add_argument('--data_root', type=str, required=True, help="Path to the real dataset root (for future resolution of relative paths if needed).")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory to save checkpoints.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Selected: {device}")
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    train_records = load_index(args.train_index)
    val_records = load_index(args.val_index)
    
    train_dataset = SARToOpticalDataset(train_records)
    val_dataset = SARToOpticalDataset(val_records)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = NoHistoryGenerator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs.")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train L1: {train_loss:.6f}")
        print(f"Val L1:   {val_loss:.6f}")
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        save_checkpoint(state, is_best, args.checkpoint_dir, filename='latest.pt')

if __name__ == "__main__":
    main()
