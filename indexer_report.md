# Temporal Indexer Test Report

## Summary Statistics
- **Number of patches processed:** 100
- **Total observations:** 3000
- **Valid cloud-free observations (<= 5%):** 137 (4.6%)

## Sequences Generated
- **Number of causal sequences (N=3):** 5
- **Number of causal sequences (N=4):** 2

## Temporal Gap Statistics (N=3)
- **Average gap:** 119.1 days
- **Min gap:** 24 days
- **Max gap:** 312 days

## History Lengths
All sequences for N=3 strictly have length 3: True

## Example Index Entry (N=3)
```json
{
  "target_index": 21,
  "target_s2_timestamp": "2018-09-10",
  "history": [
    {
      "history_index": 1,
      "days_since": 239,
      "month": 1,
      "s2_timestamp": "2018-01-14"
    },
    {
      "history_index": 8,
      "days_since": 156,
      "month": 4,
      "s2_timestamp": "2018-04-07"
    },
    {
      "history_index": 14,
      "days_since": 83,
      "month": 6,
      "s2_timestamp": "2018-06-19"
    }
  ]
}
```

## Strict Causality Check
**Violations found:** None
