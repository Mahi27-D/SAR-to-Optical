import torch
import torch.nn as nn
from .sar_encoder import SAREncoder
from .optical_encoder import OpticalEncoder
from .generator import UpBlock

class EqualWeightGenerator(nn.Module):
    """
    Generator for the Equal-Weight History Baseline.
    Uses SAREncoder for target SAR, OpticalEncoder for historical optical captures.
    Combines bottleneck features by taking the mean of history optical features,
    concatenating with SAR features, and projecting.
    """
    def __init__(self):
        super(EqualWeightGenerator, self).__init__()
        
        self.sar_encoder = SAREncoder()
        self.opt_encoder = OpticalEncoder()
        
        # Fusion projection: [B, 512 (SAR) + 512 (History), 8, 8] -> [B, 512, 8, 8]
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoding path (U-Net style, matching NoHistoryGenerator)
        # f5: [B, 512, 8, 8]
        # f4: [B, 256, 16, 16] -> skip_channels=256
        self.up4 = UpBlock(in_channels=512, out_channels=256, skip_channels=256) 
        
        # f3: [B, 128, 32, 32] -> skip_channels=128
        self.up3 = UpBlock(in_channels=256, out_channels=128, skip_channels=128)
        
        # f2: [B, 64, 64, 64] -> skip_channels=64
        self.up2 = UpBlock(in_channels=128, out_channels=64, skip_channels=64)
        
        # f1: [B, 64, 128, 128] -> skip_channels=64
        self.up1 = UpBlock(in_channels=64, out_channels=64, skip_channels=64)
        
        # Final upsample to get back to 256x256
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        # Output convolutions
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1)
        )
        
    def forward(self, sar, history):
        """
        Forward pass for the Equal-Weight Generator.
        Input:
            sar: [B, 2, 256, 256]
            history: [B, 3, 4, 256, 256]
        Output:
            out: Synthetic optical image [B, 4, 256, 256]
        """
        B, N, C, H, W = history.shape
        
        # Encode SAR
        f1_sar, f2_sar, f3_sar, f4_sar, f5_sar = self.sar_encoder(sar)
        
        # Encode History Optical
        # Flatten the history dimension to process all captures through the shared encoder
        history_flat = history.view(B * N, C, H, W)
        _, _, _, _, f5_opt_flat = self.opt_encoder(history_flat)
        
        # Reshape to [B, N, 512, 8, 8]
        _, C_f5, H_f5, W_f5 = f5_opt_flat.shape
        f5_opt = f5_opt_flat.view(B, N, C_f5, H_f5, W_f5)
        
        # Equal weighting average over the N captures
        f_history = torch.mean(f5_opt, dim=1) # [B, 512, 8, 8]
        
        # Fusion
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
        
        return out
