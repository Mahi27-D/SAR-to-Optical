import os
import json
import argparse
from datetime import datetime

from src.data.temporal_index import extract_causal_sequences
from run_real_pilot import (
    OFFICIAL_ROI, OFFICIAL_TEST_SPLIT, OFFICIAL_VAL_SPLIT,
    check_real_data_exists, extract_patch_id_from_filename,
    extract_timestamp_from_filename, load_real_geotiff,
    get_official_cloud_cover
)

def serialize_for_json(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj

def process_split(s1_root, s2_root, valid_rois, n_history, cloud_threshold):
    observations_by_patch = {}
    
    # Mirroring the robust traversal logic validated in the pilot
    for roi_path in valid_rois:
        for tdx in range(30):
            path_s1 = os.path.join(s1_root, roi_path, 'S1', str(tdx))
            path_s2 = os.path.join(s2_root, roi_path, 'S2', str(tdx))
            
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
                
                # Cloud calculation (official implementation)
                s2_img, _, _ = load_real_geotiff(s2_full, expected_bands=13)
                cloud_cover = get_official_cloud_cover(s2_img)
                
                patch_uid = f"{roi_path}_{patch_id}"
                
                # timestamps are actual python datetime objects from extract_timestamp_from_filename
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

    all_sequences = []
    for patch_uid, obs_list in observations_by_patch.items():
        obs_list.sort(key=lambda x: x['s2_timestamp'])
        seqs = extract_causal_sequences(obs_list, N=n_history, max_cloud_cover=cloud_threshold)
        all_sequences.extend(seqs)
        
    return all_sequences

def main():
    parser = argparse.ArgumentParser(description="Generate causal JSON indexes for SEN12MS-CR-TS.")
    parser.add_argument('--s1_root', type=str, required=True)
    parser.add_argument('--s2_root', type=str, required=True)
    parser.add_argument('--n_history', type=int, default=3)
    parser.add_argument('--cloud_threshold', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    if not check_real_data_exists(args.s1_root, args.s2_root):
        print("ERROR: Real SEN12MS-CR-TS data not found.")
        exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_roi = [os.path.join(k, v) for k, vals in OFFICIAL_ROI.items() for v in vals]
    train_rois = [r for r in all_roi if r not in OFFICIAL_TEST_SPLIT and r not in OFFICIAL_VAL_SPLIT]
    
    splits = {
        'train': train_rois,
        'val': OFFICIAL_VAL_SPLIT,
        'test': OFFICIAL_TEST_SPLIT
    }
    
    for split_name, rois in splits.items():
        print(f"Processing {split_name} split...")
        seqs = process_split(args.s1_root, args.s2_root, rois, args.n_history, args.cloud_threshold)
        
        output_path = os.path.join(args.output_dir, f"{split_name}_index.json")
        with open(output_path, 'w') as f:
            json.dump(serialize_for_json(seqs), f, indent=2)
        print(f"Saved {len(seqs)} sequences to {output_path}")

if __name__ == "__main__":
    main()
