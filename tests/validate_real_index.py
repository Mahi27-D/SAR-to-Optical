import sys
import json
import os
from datetime import datetime

def validate_index(index_path, expected_n=3):
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found.")
        sys.exit(1)
        
    with open(index_path, 'r') as f:
        sequences = json.load(f)
        
    failures = 0
    total = len(sequences)
    
    for i, seq in enumerate(sequences):
        try:
            # Check keys
            for key in ["target_index", "target_s1_path", "target_s2_path", "target_s1_timestamp", "target_s2_timestamp", "history"]:
                assert key in seq, f"Missing key: {key}"
                
            assert len(seq["history"]) == expected_n, f"History length {len(seq['history'])} != {expected_n}"
            
            # Paths exist
            assert os.path.exists(seq["target_s1_path"]), f"S1 path not found: {seq['target_s1_path']}"
            assert os.path.exists(seq["target_s2_path"]), f"S2 path not found: {seq['target_s2_path']}"
            
            # Parse target timestamp
            target_ts = datetime.fromisoformat(seq["target_s2_timestamp"])
            
            prev_ts = None
            
            for h in seq["history"]:
                for key in ["history_index", "days_since", "month", "s2_timestamp", "s2_path"]:
                    assert key in h, f"History missing key: {key}"
                    
                assert os.path.exists(h["s2_path"]), f"History S2 path not found: {h['s2_path']}"
                hist_ts = datetime.fromisoformat(h["s2_timestamp"])
                
                assert hist_ts < target_ts, "History timestamp not strictly less than target timestamp"
                
                days_diff = (target_ts - hist_ts).days
                assert h["days_since"] == days_diff, f"days_since mismatch: {h['days_since']} != {days_diff}"
                
                if prev_ts is not None:
                    assert prev_ts < hist_ts, "History timestamps not strictly increasing"
                prev_ts = hist_ts
                
        except AssertionError as e:
            print(f"Sequence {i} failed: {e}")
            failures += 1
            
    print(f"Total sequences checked: {total}")
    print(f"Total failures: {failures}")
    
    if failures > 0:
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_real_index.py <index_json>")
        sys.exit(1)
    validate_index(sys.argv[1])
