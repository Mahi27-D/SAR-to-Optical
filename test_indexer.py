import random
from datetime import datetime, timedelta
import json
from src.data.temporal_index import extract_causal_sequences, process_dataset_metadata

def generate_mock_data(num_patches=100):
    start_date = datetime(2018, 1, 1)
    mock_data = {}
    for patch_id in range(num_patches):
        observations = []
        for i in range(30):
            # Bi-weekly intervals roughly (12 days)
            date = start_date + timedelta(days=i * 12 + random.randint(0, 2))
            cloud_cover = random.uniform(0.0, 1.0)
            observations.append({
                'index': i,
                's2_timestamp': date,
                's1_timestamp': date + timedelta(days=random.randint(-1, 1)),
                'cloud_cover': cloud_cover
            })
        mock_data[f"patch_{patch_id}"] = observations
    return mock_data

def run_test():
    random.seed(42)
    mock_data = generate_mock_data(100) # 100 patches
    
    # 1. Total valid cloud-free observations
    total_obs = 100 * 30
    cloud_free_obs = 0
    for obs_list in mock_data.values():
        for obs in obs_list:
            if obs['cloud_cover'] <= 0.05:
                cloud_free_obs += 1
                
    # 2. Extract sequences for N=3 and N=4
    sequences_n3 = process_dataset_metadata(mock_data, N=3, max_cloud_cover=0.05)
    sequences_n4 = process_dataset_metadata(mock_data, N=4, max_cloud_cover=0.05)
    
    total_seq_n3 = sum(len(seqs) for seqs in sequences_n3.values())
    total_seq_n4 = sum(len(seqs) for seqs in sequences_n4.values())
    
    # 3. Gap statistics and history lengths (for N=3)
    history_lengths = []
    temporal_gaps = []
    
    for seqs in sequences_n3.values():
        for seq in seqs:
            history_lengths.append(len(seq['history']))
            for h in seq['history']:
                temporal_gaps.append(h['days_since'])
                
    if temporal_gaps:
        avg_gap = sum(temporal_gaps) / len(temporal_gaps)
        max_gap = max(temporal_gaps)
        min_gap = min(temporal_gaps)
    else:
        avg_gap = max_gap = min_gap = 0
        
    report = f"""# Temporal Indexer Test Report

## Summary Statistics
- **Number of patches processed:** 100
- **Total observations:** {total_obs}
- **Valid cloud-free observations (<= 5%):** {cloud_free_obs} ({(cloud_free_obs/total_obs)*100:.1f}%)

## Sequences Generated
- **Number of causal sequences (N=3):** {total_seq_n3}
- **Number of causal sequences (N=4):** {total_seq_n4}

## Temporal Gap Statistics (N=3)
- **Average gap:** {avg_gap:.1f} days
- **Min gap:** {min_gap} days
- **Max gap:** {max_gap} days

## History Lengths
All sequences for N=3 strictly have length 3: {all(l == 3 for l in history_lengths)}

## Example Index Entry (N=3)
"""
    if sequences_n3:
        example_patch = list(sequences_n3.keys())[0]
        example_seq = sequences_n3[example_patch][0]
        report += "```json\n" + json.dumps(example_seq, indent=2) + "\n```\n"
        
    report += "\n## Strict Causality Check\n"
    violation = False
    for patch, obs_list in mock_data.items():
        if patch in sequences_n3:
            for seq in sequences_n3[patch]:
                target_date = datetime.strptime(seq['target_s2_timestamp'], "%Y-%m-%d")
                for h in seq['history']:
                    # Reconstruct approx date
                    h_date = target_date - timedelta(days=h['days_since'])
                    if h_date >= target_date:
                        violation = True
    
    report += f"**Violations found:** {'Yes' if violation else 'None'}\n"
    
    with open("c:/Users/Mahi Daryan/Desktop/SAR-to-Optical/indexer_report.md", "w") as f:
        f.write(report)
        
    print("Report generated successfully.")

if __name__ == "__main__":
    run_test()
