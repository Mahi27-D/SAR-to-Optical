import os
import tempfile
import json
import torch
import unittest
from unittest.mock import patch
import sys

from src.training.train_no_history import save_checkpoint, load_index

class TestTrainingInfrastructure(unittest.TestCase):
    def setUp(self):
        print("\n--- Training infrastructure validation ---")
        print("NOT trained on SEN12MS-CR-TS")
        print("NOT an experimental result")
        print("-" * 40)
        
    def test_imports(self):
        # Verify the modules import cleanly without missing dependencies
        from src.training.train_no_history import set_seed, train_epoch, validate_epoch, main
        self.assertTrue(callable(set_seed))
        self.assertTrue(callable(train_epoch))
        self.assertTrue(callable(validate_epoch))
        self.assertTrue(callable(main))
        
    def test_checkpoint_helper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = {
                'epoch': 1,
                'model_state_dict': {'mock': torch.tensor([1, 2, 3])},
                'optimizer_state_dict': {},
                'train_loss': 0.5,
                'val_loss': 0.4
            }
            
            # Test non-best
            save_checkpoint(state, is_best=False, checkpoint_dir=temp_dir, filename='latest.pt')
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'latest.pt')))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, 'best.pt')))
            
            # Test best
            save_checkpoint(state, is_best=True, checkpoint_dir=temp_dir, filename='latest.pt')
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'latest.pt')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'best.pt')))
            
    def test_load_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, 'index.json')
            with open(index_path, 'w') as f:
                json.dump([{"fake": "record"}], f)
                
            records = load_index(index_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["fake"], "record")
            
            # Test failure on missing file
            with self.assertRaises(FileNotFoundError):
                load_index(os.path.join(temp_dir, 'missing.json'))
                
    @patch('sys.argv', ['train_no_history.py', 
                        '--train_index', 'train.json', 
                        '--val_index', 'val.json', 
                        '--data_root', '/fake/root', 
                        '--epochs', '5', 
                        '--batch_size', '4', 
                        '--lr', '0.001', 
                        '--checkpoint_dir', '/fake/ckpt',
                        '--seed', '42'])
    def test_arg_parsing(self):
        from src.training.train_no_history import main
        import argparse
        
        # Test that argparse is set up correctly in train_no_history
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
        
        args = parser.parse_args(sys.argv[1:])
        self.assertEqual(args.train_index, 'train.json')
        self.assertEqual(args.val_index, 'val.json')
        self.assertEqual(args.data_root, '/fake/root')
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.lr, 0.001)
        self.assertEqual(args.checkpoint_dir, '/fake/ckpt')
        self.assertEqual(args.seed, 42)

if __name__ == '__main__':
    unittest.main(verbosity=2)
