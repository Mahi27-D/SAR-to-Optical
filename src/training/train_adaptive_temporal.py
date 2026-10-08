import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import shutil

from src.models.adaptive_temporal_generator import AdaptiveTemporalGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.losses.perceptual import PerceptualLoss
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

def extract_metadata(history_list, device):
    """
    Extracts days_since and month from the dataloader history list.
    history_list is a list of N dictionaries.
    Returns: days_since [B, N], month [B, N]
    """
    N = len(history_list)
    if N == 0:
        raise ValueError("History list is empty.")
    B = len(history_list[0]['days_since'])
    
    days_since = torch.zeros(B, N, device=device)
    month = torch.zeros(B, N, device=device)
    
    for i in range(N):
        days_since[:, i] = history_list[i]['days_since'].to(device).float()
        month[:, i] = history_list[i]['month'].to(device).float()
        
    return days_since, month

def train_epoch(generator, discriminator, dataloader, opt_g, opt_d, criterion_l1, criterion_perceptual, device, lambda_adv=1.0, lambda_l1=100.0, lambda_perc=10.0):
    generator.train()
    discriminator.train()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        sar = batch['sar'].to(device)
        history = batch['history_optical'].to(device)
        target = batch['target_optical'].to(device)
        
        days_since, month = extract_metadata(batch['history'], device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        opt_d.zero_grad()
        
        # Real loss (Label = 1)
        pred_real = discriminator(sar, target)
        loss_d_real = 0.5 * torch.mean((pred_real - 1) ** 2)
        
        # Fake loss (Label = 0)
        with torch.no_grad():
            fake_target, _ = generator(sar, history, days_since, month)
            
        pred_fake = discriminator(sar, fake_target.detach())
        loss_d_fake = 0.5 * torch.mean((pred_fake) ** 2)
        
        loss_d = loss_d_real + loss_d_fake
        
        if not torch.isfinite(loss_d):
            raise ValueError(f"D loss is {loss_d.item()} (not finite). Stopping training.")
            
        loss_d.backward()
        opt_d.step()
        
        # ---------------------
        # Train Generator
        # ---------------------
        opt_g.zero_grad()
        
        fake_target, _ = generator(sar, history, days_since, month)
        
        # Adversarial loss (Label = 1)
        pred_fake_g = discriminator(sar, fake_target)
        loss_g_adv = 0.5 * torch.mean((pred_fake_g - 1) ** 2)
        
        # L1 and Perceptual loss
        loss_g_l1 = criterion_l1(fake_target, target)
        loss_g_perc = criterion_perceptual(fake_target, target)
        
        loss_g = (lambda_adv * loss_g_adv) + (lambda_l1 * loss_g_l1) + (lambda_perc * loss_g_perc)
        
        if not torch.isfinite(loss_g):
            raise ValueError(f"G loss is {loss_g.item()} (not finite). Stopping training.")
            
        loss_g.backward()
        opt_g.step()
        
        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()
        
    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)

@torch.no_grad()
def validate_epoch(generator, discriminator, dataloader, criterion_l1, criterion_perceptual, device, lambda_adv=1.0, lambda_l1=100.0, lambda_perc=10.0):
    generator.eval()
    discriminator.eval()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        sar = batch['sar'].to(device)
        history = batch['history_optical'].to(device)
        target = batch['target_optical'].to(device)
        
        days_since, month = extract_metadata(batch['history'], device)
        
        # D validation
        pred_real = discriminator(sar, target)
        loss_d_real = 0.5 * torch.mean((pred_real - 1) ** 2)
        
        fake_target, _ = generator(sar, history, days_since, month)
        pred_fake = discriminator(sar, fake_target)
        loss_d_fake = 0.5 * torch.mean((pred_fake) ** 2)
        
        loss_d = loss_d_real + loss_d_fake
        
        # G validation
        loss_g_adv = 0.5 * torch.mean((pred_fake - 1) ** 2)
        loss_g_l1 = criterion_l1(fake_target, target)
        loss_g_perc = criterion_perceptual(fake_target, target)
        
        loss_g = (lambda_adv * loss_g_adv) + (lambda_l1 * loss_g_l1) + (lambda_perc * loss_g_perc)
        
        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()
        
    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Temporal Baseline Training Infrastructure")
    parser.add_argument('--train_index', type=str, required=True, help="Path to train index.")
    parser.add_argument('--val_index', type=str, required=True, help="Path to val index.")
    parser.add_argument('--data_root', type=str, required=True, help="Path to data root.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size.")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Checkpoint directory.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--smoke_test', action='store_true', help="Run single mini-batch smoke test.")
    parser.add_argument('--one_epoch', action='store_true', help="Run one full epoch smoke test.")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Selected: {device}")
    
    if not args.smoke_test and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    train_records = load_index(args.train_index)
    val_records = load_index(args.val_index)
    
    if args.smoke_test:
        train_records = train_records[:args.batch_size]
        val_records = val_records[:args.batch_size]
        
    train_dataset = SARToOpticalDataset(train_records)
    val_dataset = SARToOpticalDataset(val_records)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    generator = AdaptiveTemporalGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    
    # Standard GAN settings
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)
    
    if args.smoke_test:
        print("Running one-batch smoke test...")
        g_loss, d_loss = train_epoch(generator, discriminator, train_loader, opt_g, opt_d, criterion_l1, criterion_perceptual, device)
        v_g_loss, v_d_loss = validate_epoch(generator, discriminator, val_loader, criterion_l1, criterion_perceptual, device)
        print(f"Smoke Test Train G Loss: {g_loss:.6f}, D Loss: {d_loss:.6f}")
        print(f"Smoke Test Val G Loss:   {v_g_loss:.6f}, D Loss: {v_d_loss:.6f}")
        print("One-batch smoke test completed successfully.")
        return
        
    epochs_to_run = 1 if args.one_epoch else args.epochs
    best_val_loss = float('inf')
    
    print(f"Starting training for {epochs_to_run} epochs.")
    for epoch in range(1, epochs_to_run + 1):
        g_loss, d_loss = train_epoch(generator, discriminator, train_loader, opt_g, opt_d, criterion_l1, criterion_perceptual, device)
        v_g_loss, v_d_loss = validate_epoch(generator, discriminator, val_loader, criterion_l1, criterion_perceptual, device)
        
        is_best = v_g_loss < best_val_loss
        if is_best:
            best_val_loss = v_g_loss
            
        print(f"Epoch {epoch}/{epochs_to_run}")
        print(f"Train G Loss: {g_loss:.6f}, D Loss: {d_loss:.6f}")
        print(f"Val G Loss:   {v_g_loss:.6f}, D Loss: {v_d_loss:.6f}")
        
        state = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'opt_g_state_dict': opt_g.state_dict(),
            'opt_d_state_dict': opt_d.state_dict(),
            'train_g_loss': g_loss,
            'train_d_loss': d_loss,
            'val_g_loss': v_g_loss,
            'val_d_loss': v_d_loss
        }
        save_checkpoint(state, is_best, args.checkpoint_dir, filename='latest.pt')

if __name__ == "__main__":
    main()
