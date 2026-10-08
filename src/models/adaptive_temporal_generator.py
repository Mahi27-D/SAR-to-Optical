import torch
import torch.nn as nn
import torch.nn.functional as F
from .sar_encoder import SAREncoder
from .optical_encoder import OpticalEncoder
from .generator import UpBlock

class AdaptiveTemporalGenerator(nn.Module):
    """
    Generator for the Adaptive Temporal Relevance model.
    Uses learned relevance weights for historical optical captures based on:
    - SAR mid-level features
    - Optical mid-level features
    - Days since (recency)
    - Month (seasonality)
    - Structural compatibility (cosine similarity)
    """
    def __init__(self):
        super(AdaptiveTemporalGenerator, self).__init__()
        
        self.sar_encoder = SAREncoder()
        self.opt_encoder = OpticalEncoder()
        
        # Temporal metadata embeddings
        self.days_embed = nn.Linear(1, 16)
        self.month_embed = nn.Linear(1, 16)
        
        # Temporal Relevance MLP
        # Input size: SAR (512) + Opt (512) + days_emb (16) + month_emb (16) + struct_sim (1) = 1057
        self.relevance_mlp = nn.Sequential(
            nn.Linear(1057, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Fusion projection: [B, 512 (SAR) + 512 (History), 8, 8] -> [B, 512, 8, 8]
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoding path
        self.up4 = UpBlock(in_channels=512, out_channels=256, skip_channels=256) 
        self.up3 = UpBlock(in_channels=256, out_channels=128, skip_channels=128)
        self.up2 = UpBlock(in_channels=128, out_channels=64, skip_channels=64)
        self.up1 = UpBlock(in_channels=64, out_channels=64, skip_channels=64)
        
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1)
        )
        
    def forward(self, sar, history, days_since, month):
        """
        Forward pass.
        Input:
            sar: [B, 2, 256, 256]
            history: [B, N, 4, 256, 256]
            days_since: [B, N]
            month: [B, N]
        Output:
            out: Synthetic optical image [B, 4, 256, 256]
            relevance_weights: [B, N]
        """
        B, N, C, H, W = history.shape
        
        # Encode SAR
        f1_sar, f2_sar, f3_sar, f4_sar, f5_sar = self.sar_encoder(sar)
        
        # Global pool SAR for relevance MLP
        sar_pool = F.adaptive_avg_pool2d(f5_sar, 1).view(B, 512)
        
        # Encode History Optical
        history_flat = history.view(B * N, C, H, W)
        _, _, _, _, f5_opt_flat = self.opt_encoder(history_flat)
        f5_opt = f5_opt_flat.view(B, N, 512, 8, 8)
        
        # Calculate relevance score for each history
        weights = []
        for i in range(N):
            opt_i = f5_opt[:, i, :, :, :] # [B, 512, 8, 8]
            opt_pool_i = F.adaptive_avg_pool2d(opt_i, 1).view(B, 512)
            
            # Structural compatibility (spatially preserved cosine similarity)
            sim_map = F.cosine_similarity(f5_sar, opt_i, dim=1) # [B, H, W]
            sim_i = sim_map.mean(dim=(1, 2)).unsqueeze(1) # [B, 1]
            
            # Metadata embeddings
            d_i = days_since[:, i].unsqueeze(1).float() # [B, 1]
            m_i = month[:, i].unsqueeze(1).float() # [B, 1]
            d_emb = self.days_embed(d_i)
            m_emb = self.month_embed(m_i)
            
            # Concat for MLP
            mlp_in = torch.cat([sar_pool, opt_pool_i, d_emb, m_emb, sim_i], dim=1) # [B, 1057]
            score_i = self.relevance_mlp(mlp_in) # [B, 1]
            weights.append(score_i)
            
        weights_tensor = torch.cat(weights, dim=1) # [B, N]
        
        # Softmax over the N history captures
        relevance_weights = F.softmax(weights_tensor, dim=1) # [B, N]
        
        # Adaptive weighted fusion
        f_history = torch.zeros_like(f5_sar) # [B, 512, 8, 8]
        for i in range(N):
            w_i = relevance_weights[:, i].view(B, 1, 1, 1)
            f_history += w_i * f5_opt[:, i, :, :, :]
            
        # Fusion with SAR
        f_fused = torch.cat([f5_sar, f_history], dim=1) # [B, 1024, 8, 8]
        f_fused = self.fusion_proj(f_fused) # [B, 512, 8, 8]
        
        # Decode using SAR skip connections
        d4 = self.up4(f_fused, f4_sar)
        d3 = self.up3(d4, f3_sar)
        d2 = self.up2(d3, f2_sar)
        d1 = self.up1(d2, f1_sar)
        
        # Final reconstruction to 256x256
        out = self.final_up(d1)
        out = self.final_conv(out)
        
        return out, relevance_weights
