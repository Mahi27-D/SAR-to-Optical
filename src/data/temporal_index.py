import json
from collections import defaultdict
from typing import List, Dict, Any
from datetime import datetime

def extract_causal_sequences(location_observations: List[Dict[str, Any]], N: int, max_cloud_cover: float = 0.05) -> List[Dict[str, Any]]:
    """
    Extracts all possible valid causal sequences for a single location.
    
    Args:
        location_observations: List of dicts ordered by time. Must contain:
                             [{'index', 's2_timestamp', 's1_timestamp', 'cloud_cover'}, ...]
        N: Required number of historical optical captures
        max_cloud_cover: Threshold for considering an observation "cloud-free"
    """
    valid_sequences = []
    
    for i, target_obs in enumerate(location_observations):
        
        # 1. Target optical must be cloud-free to serve as ground truth
        if target_obs['cloud_cover'] > max_cloud_cover:
            continue
            
        target_timestamp = target_obs['s2_timestamp']
            
        # 2. Filter historical candidates by strict causal invariant and cloud cover
        valid_history = []
        for h in location_observations:
            if h['s2_timestamp'] < target_timestamp and h['cloud_cover'] <= max_cloud_cover:
                valid_history.append(h)
        
        # 3. Do we have enough history?
        if len(valid_history) >= N:
            # Sort chronologically to be safe and take the N most recent
            valid_history = sorted(valid_history, key=lambda x: x['s2_timestamp'])
            selected_history = valid_history[-N:]
            
            # 4. Compute causal metadata
            history_meta = []
            for h in selected_history:
                # Explicit causal invariant assertion
                assert h['s2_timestamp'] < target_timestamp, "Violation of strict causality!"
                
                days_since = (target_timestamp - h['s2_timestamp']).days
                history_meta.append({
                    "history_index": h['index'],
                    "days_since": days_since,
                    "month": h['s2_timestamp'].month,
                    "s2_timestamp": h['s2_timestamp'].strftime("%Y-%m-%d"),
                })
                
            valid_sequences.append({
                "target_index": target_obs['index'],
                "target_s2_timestamp": target_timestamp.strftime("%Y-%m-%d"),
                "target_s1_timestamp": target_obs.get('s1_timestamp').strftime("%Y-%m-%d") if 's1_timestamp' in target_obs else None,
                "target_s2_path": target_obs.get('s2_path'),
                "target_s1_path": target_obs.get('s1_path'),
                "history": history_meta
            })
            
    return valid_sequences

def process_dataset_metadata(patch_metadata: Dict[str, List[Dict[str, Any]]], N: int, max_cloud_cover: float = 0.05) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a collection of patches to generate causal temporal sequences.
    
    Args:
        patch_metadata: A dictionary mapping patch_id to a list of its 30 observations.
    """
    sequences_by_patch = {}
    
    for patch_id, observations in patch_metadata.items():
        # Ensure observations are sorted by s2_timestamp
        obs_sorted = sorted(observations, key=lambda x: x['s2_timestamp'])
        sequences = extract_causal_sequences(obs_sorted, N, max_cloud_cover)
        if sequences:
            sequences_by_patch[patch_id] = sequences
            
    return sequences_by_patch
