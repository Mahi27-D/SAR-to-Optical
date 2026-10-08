import torch
import torch.nn as nn
import torchvision.models as models

class OpticalEncoder(nn.Module):
    """
    Optical Encoder based on ResNet-18 architecture.
    Extracts multi-scale features for historical optical captures.
    """
    def __init__(self):
        super(OpticalEncoder, self).__init__()
        
        # Load ResNet18 without pretrained weights
        resnet = models.resnet18(weights=None)
        
        # Modify the first convolution to accept 4 channels (B02, B03, B04, B08) instead of 3
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Residual blocks
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(self, x):
        """
        Forward pass extracting multi-scale features.
        Input:
            x: Optical image [B, 4, 256, 256]
        Outputs:
            f1: [B, 64, 128, 128]
            f2: [B, 64, 64, 64]
            f3: [B, 128, 32, 32]
            f4: [B, 256, 16, 16]
            f5: [B, 512, 8, 8]
        """
        # Feature level 1
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        f1 = self.relu(x1)
        
        # Feature level 2
        x2 = self.maxpool(f1)
        f2 = self.layer1(x2)
        
        # Feature level 3
        f3 = self.layer2(f2)
        
        # Feature level 4
        f4 = self.layer3(f3)
        
        # Feature level 5
        f5 = self.layer4(f4)
        
        return f1, f2, f3, f4, f5
