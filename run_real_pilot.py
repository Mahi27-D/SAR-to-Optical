import os
import argparse
import glob
from datetime import datetime
import json
import warnings

import tifffile
import numpy as np

try:
    from s2cloudless import S2PixelCloudDetector
except ImportError:
    S2PixelCloudDetector = None

from src.data.temporal_index import extract_causal_sequences

# Official splits from SEN12MS-CR-TS (dataLoader.py)
OFFICIAL_ROI = {
    'ROIs1158': ['106'],
    'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
    'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
    'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146']
}

OFFICIAL_TEST_SPLIT = [
    os.path.join('ROIs1868', '119'), os.path.join('ROIs1970', '139'), os.path.join('ROIs2017', '108'), 
    os.path.join('ROIs2017', '63'), os.path.join('ROIs1158', '106'), os.path.join('ROIs1868', '73'), 
    os.path.join('ROIs2017', '32'), os.path.join('ROIs1868', '100'), os.path.join('ROIs1970', '132'), 
    os.path.join('ROIs2017', '103'), os.path.join('ROIs1868', '142'), os.path.join('ROIs1970', '20'), 
    os.path.join('ROIs2017', '140')
]

OFFICIAL_VAL_SPLIT = [
    os.path.join('ROIs2017', '22'), os.path.join('ROIs1970', '65'), os.path.join('ROIs2017', '117'), 
    os.path.join('ROIs1868', '127'), os.path.join('ROIs1868', '17')
]

def check_real_data_exists(s1_root, s2_root):
    if not os.path.exists(s1_root) or not os.path.isdir(s1_root) or \
       not os.path.exists(s2_root) or not os.path.isdir(s2_root):
        return False
    
    tif_files = glob.glob(os.path.join(s1_root, '**', 'S1', '**', '*.tif'), recursive=True)
    if not tif_files:
        return False
    return True

def get_official_cloud_cover(s2_img):
    """
    Reproduces the official s2cloudless_mask mechanism from SEN12MS-CR-TS.
    Requires s2cloudless. Does NOT use brightness heuristics or sidecar files.
    """
    if S2PixelCloudDetector is None:
        raise ImportError("s2cloudless is not installed. Please install it to run the official cloud mask logic.")
        
    # Clip and normalize as per official loader
    intensity_min, intensity_max = 0, 10000
    img = np.clip(s2_img, intensity_min, intensity_max)
    img = img / 10000.0
    
    # Needs channel-last for s2cloudless, but img is (13, H, W)
    # The detector expects (1, H, W, 13) or (H, W, 13) usually if processing a single image.
    img_channels_last = np.transpose(img, (1, 2, 0))
    # Note: s2cloudless expects a batch dim, so (1, H, W, 13) or list of images.
    img_batch = np.expand_dims(img_channels_last, axis=0)
    
    detector = S2PixelCloudDetector(
        threshold=0.4,
        all_bands=True,
        average_over=4,
        dilation_size=2
    )
    
    mask = detector.get_cloud_masks(img_batch)[0]
    return float(np.mean(mask))

def load_real_geotiff(path, expected_bands=None):
    """
    Loads TIFF and explicitly validates spatial dimension and expected bands.
    """
    img = tifffile.imread(path)
    
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0) 
    elif len(img.shape) == 3 and img.shape[2] <= 13:
        # Transpose if the TIFF was saved channel-last
        img = np.transpose(img, (2, 0, 1))
        
    bands, height, width = img.shape
    
    if expected_bands and bands != expected_bands:
        raise ValueError(f"Expected {expected_bands} bands, but got {bands} in {path}")
        
    if height != 256 or width != 256:
        raise ValueError(f"Expected spatial dimensions 256x256, got {height}x{width} in {path}")
        
    return img, img.dtype, (bands, height, width)

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

def extract_timestamp_from_filename(filename):
    """
    Official SEN12MS-CR-TS structure embeds the timestamp at split index 5.
    e.g., ROIs1158_spring_s1_1_p1_2018-05-15.tif
    """
    parts = os.path.basename(filename).split('_')
    if len(parts) < 6:
        raise ValueError(f"Filename {filename} does not match official SEN12MS-CR-TS timestamp format.")
    date_str = parts[5].replace('.tif', '')
    return datetime.strptime(date_str, '%Y-%m-%d')

def extract_patch_id_from_filename(filename):
    """
    Extracts the patch identifier after '_patch_'
    (e.g. 's1_ROIs..._patch_113.tif' -> '113').
    """
    return os.path.basename(filename).split('_patch_')[-1].replace('.tif', '')

def main():
    parser = argparse.ArgumentParser(description="Real-data pilot validation for SEN12MS-CR-TS.")
    parser.add_argument('--s1_root', type=str, required=True, help="Path to real S1 dataset root.")
    parser.add_argument('--s2_root', type=str, required=True, help="Path to real S2 dataset root.")
    parser.add_argument('--n_history', type=int, default=3, help="Number of historical optical observations required.")
    parser.add_argument('--cloud_threshold', type=float, default=0.05, help="Project cloud cover sequence-validity threshold (not internal s2cloudless 0.4).")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'])
    
    args = parser.parse_args()
    
    if not check_real_data_exists(args.s1_root, args.s2_root):
        print("\nERROR: No real SEN12MS-CR-TS data found. Synthetic validation is not real-data validation.")
        print("Stopping pilot script.")
        exit(1)
        
    # Build list of valid ROIs based on official splits
    valid_rois = []
    if args.split == 'test':
        valid_rois = OFFICIAL_TEST_SPLIT
    elif args.split == 'val':
        valid_rois = OFFICIAL_VAL_SPLIT
    elif args.split == 'train':
        all_roi = [os.path.join(k, v) for k, vals in OFFICIAL_ROI.items() for v in vals]
        valid_rois = [r for r in all_roi if r not in OFFICIAL_TEST_SPLIT and r not in OFFICIAL_VAL_SPLIT]
    else:
        valid_rois = [os.path.join(k, v) for k, vals in OFFICIAL_ROI.items() for v in vals]
        
    observations_by_patch = {}
    
    for roi_path in valid_rois:
        for tdx in range(30):
            path_s1 = os.path.join(args.s1_root, roi_path, 'S1', str(tdx))
            path_s2 = os.path.join(args.s2_root, roi_path, 'S2', str(tdx))
            
            if not os.path.exists(path_s1) or not os.path.exists(path_s2):
                continue
                
            s1_files = sorted([f for f in os.listdir(path_s1) if f.endswith('.tif')])
            s2_files = sorted([f for f in os.listdir(path_s2) if f.endswith('.tif')])
            
            s1_dict = {extract_patch_id_from_filename(f): f for f in s1_files}
            s2_dict = {extract_patch_id_from_filename(f): f for f in s2_files}
            
            if set(s1_dict.keys()) != set(s2_dict.keys()):
                raise ValueError(f"S1 and S2 patch ID sets differ at time index {tdx} in {roi_path}")
                
            for patch_id in s1_dict.keys():
                s1_f = s1_dict[patch_id]
                s2_f = s2_dict[patch_id]
                
                s1_full = os.path.join(path_s1, s1_f)
                s2_full = os.path.join(path_s2, s2_f)
                
                s1_date = extract_timestamp_from_filename(s1_f)
                s2_date = extract_timestamp_from_filename(s2_f)
                
                # Assert that S1 and S2 correspond to the same patch ID and temporal index
                assert extract_patch_id_from_filename(s1_f) == extract_patch_id_from_filename(s2_f), "Target patch mismatch"
                # Since they are located in the same tdx folder, the temporal index is the same.
                
                # Official validation:
                s2_img, _, _ = load_real_geotiff(s2_full, expected_bands=13)
                s1_img, _, _ = load_real_geotiff(s1_full, expected_bands=2)
                
                s2_subset = extract_optical_bands(s2_img)
                cloud_cover = get_official_cloud_cover(s2_img)
                
                patch_uid = f"{roi_path}_{patch_id}"
                
                obs = {
                    'index': tdx,
                    's2_timestamp': s2_date,
                    's1_timestamp': s1_date,
                    'cloud_cover': cloud_cover,
                    's2_path': s2_full,
                    's1_path': s1_full
                }
                
                if patch_uid not in observations_by_patch:
                    observations_by_patch[patch_uid] = []
                observations_by_patch[patch_uid].append(obs)

    # Sort and extract sequences
    total_sequences = 0
    all_sequences = []
    
    for patch_uid, obs_list in observations_by_patch.items():
        obs_list.sort(key=lambda x: x['s2_timestamp'])
        seqs = extract_causal_sequences(obs_list, N=args.n_history, max_cloud_cover=args.cloud_threshold)
        total_sequences += len(seqs)
        all_sequences.extend(seqs)
        
        for seq in seqs:
            target_date = datetime.strptime(seq['target_s2_timestamp'], "%Y-%m-%d")
            assert len(seq['history']) == args.n_history
            for h in seq['history']:
                h_date = datetime.strptime(h['s2_timestamp'], "%Y-%m-%d")
                assert h_date < target_date, "Strict causality violation"
                
                # Verify historical observation was indeed cloud-free relative to project sequence validity threshold
                hist_obs = next(o for o in obs_list if o['index'] == h['history_index'])
                assert hist_obs['cloud_cover'] <= args.cloud_threshold, "Historical observation exceeds project cloud threshold"
                
    print(f"Total sequences generated: {total_sequences}")

if __name__ == "__main__":
    main()
