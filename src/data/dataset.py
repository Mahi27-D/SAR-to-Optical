import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
import os

def load_geotiff(path, expected_bands=None):
    img = tifffile.imread(path)
    
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0) 
    elif len(img.shape) == 3 and img.shape[2] <= 13:
        img = np.transpose(img, (2, 0, 1))
        
    bands, height, width = img.shape
    if expected_bands and bands != expected_bands:
        raise ValueError(f"Expected {expected_bands} bands, but got {bands} in {path}")
        
    return img

def extract_optical_bands(s2_img):
    """
    Extract B02, B03, B04, B08 from Sentinel-2 13-band image (0-indexed).
    B02=1, B03=2, B04=3, B08=7
    """
    if s2_img.shape[0] < 13:
        raise ValueError(f"Expected 13 bands for S2, found {s2_img.shape[0]}")
    b02 = s2_img[1, :, :]
    b03 = s2_img[2, :, :]
    b04 = s2_img[3, :, :]
    b08 = s2_img[7, :, :]
    return np.stack([b02, b03, b04, b08], axis=0)

class SARToOpticalDataset(Dataset):
    """
    PyTorch Dataset consuming precomputed causal sequence records.
    Exposes current target-time SAR and target optical ground truth, alongside historical metadata.
    """
    def __init__(self, causal_sequences):
        """
        Args:
            causal_sequences: List of dictionaries representing causal sequences.
        """
        self.sequences = causal_sequences
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        target_s1_path = seq.get('target_s1_path')
        target_s2_path = seq.get('target_s2_path')
        
        if not target_s1_path or not os.path.exists(target_s1_path):
            raise FileNotFoundError(f"SAR path missing or invalid: {target_s1_path}")
            
        if not target_s2_path or not os.path.exists(target_s2_path):
            raise FileNotFoundError(f"Optical path missing or invalid: {target_s2_path}")
            
        # Load SAR (expected 2 bands: VV, VH)
        sar_img = load_geotiff(target_s1_path, expected_bands=2)
        sar_tensor = torch.from_numpy(sar_img.astype(np.float32))
        
        # Load Optical (expected 13 bands)
        s2_img = load_geotiff(target_s2_path, expected_bands=13)
        optical_subset = extract_optical_bands(s2_img)
        optical_tensor = torch.from_numpy(optical_subset.astype(np.float32))
        
        history_tensors = []
        for h in seq.get('history', []):
            h_path = h.get('s2_path')
            if not h_path or not os.path.exists(h_path):
                raise FileNotFoundError(f"History optical path missing or invalid: {h_path}")
            h_img = load_geotiff(h_path, expected_bands=13)
            h_subset = extract_optical_bands(h_img)
            history_tensors.append(torch.from_numpy(h_subset.astype(np.float32)))
            
        history_optical_tensor = torch.stack(history_tensors, dim=0) if history_tensors else torch.empty(0)

        sample = {
            "sar": sar_tensor,
            "target_optical": optical_tensor,
            "history_optical": history_optical_tensor,
            "target_s1_timestamp": seq.get('target_s1_timestamp'),
            "target_s2_timestamp": seq.get('target_s2_timestamp'),
            "target_index": seq.get('target_index'),
            "history": seq.get('history', [])
        }
        
        return sample
