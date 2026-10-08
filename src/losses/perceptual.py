import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    """
    Computes VGG19 perceptual loss on the RGB channels.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        # x, y: [B, 4, H, W] - take only first 3 channels (RGB)
        x = x[:, :3, :, :]
        y = y[:, :3, :, :]
        
        # We assume x and y are scaled roughly to match VGG input bounds,
        # but since we just compute relative L1 loss on features, basic [-1, 1] works fine.
        h_x = self.slice1(x)
        h_y = self.slice1(y)
        loss = nn.functional.l1_loss(h_x, h_y)
        
        h_x = self.slice2(h_x)
        h_y = self.slice2(h_y)
        loss += nn.functional.l1_loss(h_x, h_y)
        
        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        loss += nn.functional.l1_loss(h_x, h_y)
        
        h_x = self.slice4(h_x)
        h_y = self.slice4(h_y)
        loss += nn.functional.l1_loss(h_x, h_y)
        
        h_x = self.slice5(h_x)
        h_y = self.slice5(h_y)
        loss += nn.functional.l1_loss(h_x, h_y)
        
        return loss
