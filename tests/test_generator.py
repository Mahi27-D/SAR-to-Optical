import torch
import torch.nn as nn
from src.models import NoHistoryGenerator

def test_generator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA device available: {torch.cuda.is_available()} | Using device: {device}")
    
    # Initialize the generator
    model = NoHistoryGenerator().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create dummy input: [Batch, Channels(2), Height, Width]
    x = torch.randn(2, 2, 256, 256, device=device)
    print(f"Input shape: {x.shape}")
    
    # Create dummy target: [Batch, Channels(4), Height, Width]
    y = torch.randn(2, 4, 256, 256, device=device)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 4, 256, 256), f"Expected shape (2, 4, 256, 256), got {output.shape}"
    
    # Simple L1 loss
    criterion = nn.L1Loss()
    loss = criterion(output, y)
    print(f"L1 Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Verify gradients
    gradients_exist = True
    gradients_finite = True
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                gradients_exist = False
                print(f"Missing gradient for {name}")
                break
            if not torch.isfinite(param.grad).all():
                gradients_finite = False
                print(f"Non-finite gradient found for {name}")
                break
                
    if gradients_exist and gradients_finite:
        print("Gradients exist and are finite: True")
    else:
        print("Gradients check failed!")

if __name__ == "__main__":
    test_generator()
