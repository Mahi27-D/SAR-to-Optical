import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for 256x256 images.
    Receptive field is 70x70.
    Inputs are concatenated SAR (2 channels) and Optical (4 channels) = 6 channels total.
    """
    def __init__(self, in_channels=6):
        super(PatchGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1: [B, 64, 128, 128]
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: [B, 128, 64, 64]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: [B, 256, 32, 32]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: [B, 512, 31, 31]
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output Layer: [B, 1, 30, 30]
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, sar, optical):
        # Concatenate SAR and Optical along the channel dimension
        img_input = torch.cat([sar, optical], dim=1)
        return self.model(img_input)
