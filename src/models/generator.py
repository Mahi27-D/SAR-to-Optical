import torch
import torch.nn as nn
from .sar_encoder import SAREncoder

class UpBlock(nn.Module):
    """
    Standard U-Net decoding block with skip connections.
    """
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # After upsampling, we concatenate with skip connection
        conv_in = (in_channels // 2) + skip_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class NoHistoryGenerator(nn.Module):
    """
    U-Net style generator for the No-History Baseline.
    Uses SAREncoder as the backbone.
    Outputs 4 channels corresponding to B02, B03, B04, B08.
    """
    def __init__(self):
        super(NoHistoryGenerator, self).__init__()
        
        self.encoder = SAREncoder()
        
        # Decoding path (U-Net style)
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
        
    def forward(self, x):
        """
        Forward pass for the No-History Baseline Generator.
        Input:
            x: SAR image [B, 2, 256, 256]
        Output:
            out: Synthetic optical image [B, 4, 256, 256]
        """
        # Encode
        f1, f2, f3, f4, f5 = self.encoder(x)
        
        # Decode
        d4 = self.up4(f5, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        
        # Final reconstruction to 256x256
        out = self.final_up(d1)
        out = self.final_conv(out)
        
        return out
