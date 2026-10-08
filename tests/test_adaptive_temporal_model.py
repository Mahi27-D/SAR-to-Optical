import torch
import unittest
from src.models.adaptive_temporal_generator import AdaptiveTemporalGenerator

class TestAdaptiveTemporalGenerator(unittest.TestCase):
    def test_forward_backward_pass(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\\nRunning test on device: {device}")
        
        model = AdaptiveTemporalGenerator().to(device)
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        
        B = 2
        N = 3
        
        # Synthetic inputs
        sar = torch.randn(B, 2, 256, 256).to(device)
        history = torch.randn(B, N, 4, 256, 256).to(device)
        days_since = torch.randint(1, 365, (B, N)).to(device)
        month = torch.randint(1, 13, (B, N)).to(device)
        
        target = torch.randn(B, 4, 256, 256).to(device)
        
        # Forward pass
        out, relevance_weights = model(sar, history, days_since, month)
        
        print(f"Output shape: {out.shape}")
        self.assertEqual(out.shape, (B, 4, 256, 256), "Output shape is incorrect")
        
        print(f"Relevance weights shape: {relevance_weights.shape}")
        self.assertEqual(relevance_weights.shape, (B, N), "Relevance weights shape is incorrect")
        
        print(f"Example relevance weights for batch 0: {relevance_weights[0].tolist()}")
        
        # Verify row sums
        row_sums = torch.sum(relevance_weights, dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(B).to(device)), f"Row sums are not 1: {row_sums.tolist()}")
        
        # Verify weights are finite and non-negative
        self.assertTrue(torch.isfinite(relevance_weights).all(), "Relevance weights contain NaN or Inf")
        self.assertTrue((relevance_weights >= 0).all(), "Relevance weights contain negative values")
        
        # Check finite values in output
        self.assertTrue(torch.isfinite(out).all(), "Output contains NaN or Inf")
        
        # Loss computation
        criterion = torch.nn.L1Loss()
        loss = criterion(out, target)
        
        # Check finite loss
        self.assertTrue(torch.isfinite(loss).all(), "Loss is NaN or Inf")
        
        # Backward pass
        loss.backward()
        
        # Verify gradients are computed and finite
        has_grads = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grads = True
                self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf")
                
        self.assertTrue(has_grads, "No gradients computed during backward pass")
        
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Peak VRAM usage: {peak_vram:.2f} MB")
            
        print("Forward and backward passes completed successfully.")

if __name__ == '__main__':
    unittest.main()
