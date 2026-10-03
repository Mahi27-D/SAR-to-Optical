# Temporal-Guided SAR-to-Optical Satellite Image Translation

## Final-Year Project — Architecture & Implementation Specification (v2.1, scoped for a November deadline)

**Purpose:** Build specification for an AI coding agent (Antigravity / Claude Code) to implement the project from an empty repository to a working, evaluable system. This version keeps the strongest ideas from the earlier draft — dataset validation first, honest claims, explainability, failure analysis — but trims the experimental workload down to what is realistically finishable in ~12 weeks, with everything else clearly marked as optional.

---

## 1. Project Overview

### 1.1 Core Goal

Given:
1. A **current SAR image** of a location.
2. A small set of **historical cloud-free optical images** of the same location.
3. Temporal metadata for those historical observations (how old, what season).

Generate a realistic **synthetic current optical image** — without ever using the true current optical image as input. The true current optical image is used **only as ground truth during training/evaluation**, never at inference.

### 1.2 Central Research Problem

SAR sees through clouds but is hard to interpret. Historical optical imagery is easy to interpret but not equally useful — usefulness depends on how long ago it was captured, what season it's from, and whether the scene has changed since. Naively averaging historical images, or using only the most recent one, ignores this.

### 1.3 Research Hypothesis

> A SAR-conditioned model that learns to adaptively weight historical optical observations — based on recency, seasonal similarity, and structural agreement with the current SAR scene — will produce more accurate and realistic reconstructions than approaches that ignore history or weight it naively.

### 1.4 Core Research Contribution (single, unified — not split into multiple "contributions")

One module: the **Temporal Relevance Module**. It takes each historical optical observation plus the current SAR features and outputs a relevance weight, using three signals fused together inside the same module:
- **Recency** (temporal distance)
- **Seasonal similarity** (month/season alignment)
- **Structural compatibility with SAR** (does this historical capture's structure agree with what SAR shows now — this also functions as a lightweight scene-change signal: low compatibility ⇒ likely stale/changed scene)

Keeping this as **one module with three signals**, rather than three separate "contributions" (temporal module + structural module + change-detection module), keeps the story tight and defensible: one clear idea, not three unproven ones stacked together.

---

## 2. Scope and Project Philosophy

This is a research/experimental project, not a product. Priorities, in order:
1. A validated, working data pipeline.
2. A working baseline model.
3. A working proposed model that measurably outperforms baselines.
4. Clear evaluation and honest reporting of results.

Anything beyond this (extra baselines, ablation grids, uncertainty maps, a demo UI) is valuable **only if time remains** after the above is solid. Do not let any of it delay the core comparison.

**Explicitly out of scope:** frontend/backend web app, LLM integration, database, deployment. This stays a training/evaluation research repository — scripts and notebooks, not a product.

---

## 3. High-Level Architecture

```
                    CURRENT SAR IMAGE
                          |
                          v
                  +----------------+
                  |  SAR Encoder   |   ResNet-18 (2-channel input: VV, VH)
                  +-------+--------+
                          |
                   current SAR features (f_sar)
                          |
      +-------------------+-------------------+
      |                   |                   |
      v                   v                   v
 Historical Opt-1   Historical Opt-2   Historical Opt-N
      |                   |                   |
      v                   v                   v
  Optical Encoder    Optical Encoder    Optical Encoder     (shared weights)
      |                   |                   |
      +-------------------+-------------------+
                          |
                          v
          +--------------------------------------+
          |     TEMPORAL RELEVANCE MODULE         |
          |  signals per historical capture:      |
          |   - days_since (recency)              |
          |   - month/season                      |
          |   - structural compatibility w/ SAR   |
          +-----------------+----------------------+
                            |
                     relevance scores → softmax
                            |
                       weights w_1..w_N
                            |
                            v
                 ADAPTIVE WEIGHTED FUSION
                    f_fused = Σ w_i · f_opt_i  (combined with f_sar)
                            |
                            v
                   U-Net Generator
              (skip connections from SAR encoder)
                            |
                            v
              SYNTHESIZED OPTICAL IMAGE
                            |
              +-------------+--------------+
              v                            v
      PatchGAN Discriminator      Loss computation
      (real vs. fake patches)   (L1 + adversarial + perceptual)
```

---

## 4. Dataset and Data Contract

### 4.1 Dataset

**SEN12MS** — paired Sentinel-1 SAR and Sentinel-2 optical patches, globally sampled.

### 4.2 Phase 0 — Dataset Feasibility Check (do this before writing any model code)

Before building the pipeline, verify:
1. Location IDs and timestamps are usable to group observations by location.
2. Each location has enough repeated optical observations over time to form a historical sequence (target: at least 4-5 per location for a usable subset).
3. Visualize a few examples: SAR + historical optical sequence + target optical, side by side, to confirm the temporal structure is real and not an artifact of how the data is indexed.

**If SEN12MS cannot reliably support this temporal structure for enough locations, stop and address this before proceeding** — e.g. scope to a regional subset with denser revisit data, or supplement via the Copernicus Open Access Hub / Earth Engine for extra historical captures. Do not proceed to full model-building on an unvalidated data assumption.

### 4.3 Training Sample

```python
Sample = {
    "sar_image": Tensor[2, 256, 256],            # VV, VH
    "past_optical": List[Tensor[4, 256, 256]],   # RGB + NIR, N=3-4 past captures
    "past_metadata": [
        {"days_since": int, "month": int, "capture_id": str}
    ],
    "target_optical": Tensor[4, 256, 256],        # training/evaluation only
    "location_id": str,
}
```

### 4.4 Preprocessing
1. Read SAR/optical bands via `rasterio`/GDAL.
2. Normalize SAR and optical bands separately, using training-split statistics.
3. Extract 256×256 patches; discard corrupted/majority-missing patches.
4. Build a location → chronologically sorted observation index.
5. For each sample, retrieve the N most recent past optical captures strictly before the target timestamp, and compute `days_since`/`month`.

### 4.5 Splits

Split **by location**, not by patch, to avoid leakage. 70% train / 15% val / 15% test — no location in more than one split.

---

## 5. Model Specification

### 5.1 SAR Encoder
ResNet-18 backbone, first conv modified to accept 2 channels (VV, VH). Outputs multi-scale feature maps (for U-Net skip connections) plus a pooled global feature vector.

### 5.2 Optical Encoder
Same backbone family, 4-channel input (RGB+NIR), weights **shared** across all historical captures for consistency and smaller model size.

### 5.3 Temporal Relevance Module (core contribution)

For each historical capture *i*:
```
inputs: f_sar, f_opt_i, embed(days_since_i), embed(month_i), structural_similarity_i
structural_similarity_i = cosine_similarity(SAR_mid_features, Optical_i_mid_features)  # resized to match

concat(all inputs) → MLP (2-3 layers, ReLU) → relevance score s_i
softmax(s_1, ..., s_N) → weights w_1, ..., w_N
```

Make the structural-similarity input configurable (on/off) so it can be turned off for a simple ablation later if time allows, without needing a separate model.

### 5.4 Adaptive Fusion
```
f_history = Σ_i (w_i · f_opt_i)
f_fused = project(concat(f_sar, f_history))
```

### 5.5 Generator
U-Net decoder, bottleneck = `f_fused`, skip connections from the **SAR encoder** (SAR is always spatially available for the current scene; historical optical is not guaranteed to be pixel-aligned). Output: 4-channel image, tanh activation.

### 5.6 Discriminator
PatchGAN (70×70 receptive field), input = concat(SAR, candidate optical image). Real pair uses ground truth optical; fake pair uses generator output.

---

## 6. Baselines (trimmed to three — keeps the comparison clean and finishable)

| # | Name | Description |
|---|---|---|
| 1 | **No-history** | SAR encoder → generator directly. No historical data, no temporal module. |
| 2 | **Equal-weight** | Temporal Relevance Module replaced with fixed `w_i = 1/N`. |
| 3 | **Proposed (adaptive)** | Full Temporal Relevance Module as specified in §5.3. |

Implement all three sharing the same encoder/generator/discriminator code, differing only in the fusion mechanism, so the comparison is fair and the codebase stays small.

**If time allows after these three are solid, optionally add:** a recency-only baseline (`w_i` as a function of `days_since` only, no learned MLP) — useful for a stronger ablation story, but not required for the core result.

---

## 7. Loss Functions

| Loss | Purpose | Starting weight |
|---|---|---|
| Adversarial (LSGAN) | Realism | 1.0 |
| L1 reconstruction | Pixel fidelity vs. ground truth | 100.0 |
| Perceptual (VGG19 features) | Texture/structure realism | 10.0 |

`L_G = λ_adv · L_adv + λ_L1 · L_L1 + λ_perc · L_perc`. Treat weights as tunable; log if you adjust them.

---

## 8. Training Procedure

1. Config-driven (YAML): model variant, data paths, hyperparameters.
2. Dataloader: batch size 8-12 (256×256 patches).
3. Adam optimizer, lr=2e-4, β1=0.5 (standard GAN settings), mixed precision (`torch.cuda.amp`) for speed on Colab/Kaggle GPUs.
4. Per epoch: train discriminator on real/fake pairs → train generator (adversarial + L1 + perceptual) → log to Weights & Biases.
5. Validate every epoch: PSNR/SSIM on val split, save sample generated images.
6. Checkpoint best model (by val PSNR or FID).

### Example config
```yaml
model:
  variant: "adaptive"          # no_history | equal_weight | adaptive
  encoder_backbone: "resnet18"
  num_past_captures: 4
  use_structural_similarity: true

data:
  patch_size: 256
  batch_size: 12

training:
  epochs: 60
  lr: 0.0002
  beta1: 0.5
  loss_weights:
    adversarial: 1.0
    l1: 100.0
    perceptual: 10.0
```

---

## 9. Evaluation

### 9.1 Standard metrics
PSNR, SSIM (pixel/structural fidelity), FID (perceptual realism, via `pytorch-fid`). Report all three — never a single metric alone.

### 9.2 Condition-based breakdown (this is your strongest result — keep it, it's cheap to compute)
Bucket test samples by **temporal gap** (e.g. 0-30 / 30-90 / 90+ days) and compare no-history vs. equal-weight vs. adaptive in each bucket. This directly demonstrates *when and why* the adaptive model helps — the core empirical claim of the project.

### 9.3 Explainability (lightweight, worth keeping)
For a handful of test samples, plot the historical captures alongside their learned relevance weights next to the generated output:
```
Capture      Days Old     Weight
Capture 1       20         0.61
Capture 2       65         0.23
Capture 3      150         0.10
Capture 4      300         0.06
```
This is mostly plotting code on top of a model you already have — low cost, strong payoff in a viva.

### 9.4 Failure case analysis (lightweight, worth keeping)
Include 3-5 failure examples (large temporal gap, seasonal mismatch, conflicting history) alongside the successes. Shows honest, complete evaluation rather than cherry-picked results.

---

## 10. Explicitly Deferred / Optional (do not build unless the above is done and time remains)

- **Additional baselines** (recency-only, latest-only)
- **Full ablation grid** (temporal-only / +seasonal / +structural, as separate trained models)
- **Standalone scene-change/uncertainty module** — the structural-compatibility signal inside the Temporal Relevance Module already gives you a lightweight version of this; don't build a separate network for it unless there's clear time and justification
- **Confidence/uncertainty output map**
- **Demo interface** (Streamlit/Gradio) — only after the research pipeline and report are done

Mark these clearly as future work in the final report if not completed — this is a legitimate and expected part of a scoped research project, not a weakness.

---

## 11. Repository Structure

```
sar-optical-translation/
├── configs/
│   ├── baseline_no_history.yaml
│   ├── baseline_equal_weight.yaml
│   └── proposed_adaptive.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── temporal_index.py
│   ├── models/
│   │   ├── encoders.py
│   │   ├── temporal_relevance.py     # ★ core contribution
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── baselines.py
│   ├── losses/
│   │   ├── adversarial.py
│   │   ├── reconstruction.py
│   │   └── perceptual.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       ├── logging.py
│       ├── metrics.py
│       └── visualize.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_temporal_validation.ipynb
│   ├── 03_baseline_results.ipynb
│   └── 04_final_comparison.ipynb
├── results/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
├── requirements.txt
├── README.md
└── run_all_baselines.sh
```

---

## 12. Recommended Build Order

| Phase | Weeks | Task |
|---|---|---|
| 0 | 1 | Dataset feasibility check (§4.2) — do not skip |
| 1 | 1-2 | Data pipeline: preprocessing, temporal index, location-based splits, dataset loader |
| 2 | 3-4 | No-history baseline (SAR encoder + generator + discriminator) — get this training and producing plausible output first |
| 3 | 5 | Equal-weight baseline — small delta on top of the working pipeline |
| 4 | 6-8 | Temporal Relevance Module (core contribution) — integrate once the rest is proven to work |
| 5 | 9-10 | Evaluation: PSNR/SSIM/FID, condition-based breakdown, explainability plots |
| 6 | 10-11 | Failure case analysis |
| 7 | 11-12 | Writeup, figures, polish, presentation prep |
| — | if time allows | Optional items from §10 |

---

## 13. Dependencies

```
torch>=2.0
torchvision
rasterio
gdal
numpy
scikit-image
pytorch-fid
wandb
pyyaml
matplotlib
seaborn
tqdm
```

---

## 14. What Not to Claim

- Do not claim this is the first system to use temporal/historical information in SAR-to-optical translation.
- Do not claim superiority without the experimental comparison (§6) to back it.
- Do not call the structural-compatibility signal "scene-change detection" unless it's validated against actual change labels — describe it as a compatibility/relevance signal, which is what it actually is.
- Clearly separate, in the final report: **what was implemented**, **what the experiments showed**, and **what is future work**.

---

## 15. Success Criteria

The project is complete and defensible if:
1. Dataset temporal structure is validated (§4.2).
2. All three baselines (§6) train successfully and reproducibly.
3. The proposed model shows a measurable improvement over both baselines on at least one metric, in at least some conditions.
4. The condition-based breakdown (§9.2) shows *where* the improvement comes from.
5. Relevance weights are visualized for a handful of examples.
6. Failure cases are honestly documented, not hidden.
7. The report clearly separates demonstrated results from future work.

## 16. Priority Rule for the Coding Agent

```
MUST HAVE                          OPTIONAL (only if time remains)
----------                         ---------------------------------
Dataset validation                 Recency-only / latest-only baselines
Data pipeline                      Full ablation grid
No-history baseline                Standalone scene-change module
Equal-weight baseline              Confidence/uncertainty map
Temporal Relevance Module          Demo interface (Streamlit/Gradio)
Adaptive fusion + U-Net + PatchGAN
PSNR / SSIM / FID
Condition-based breakdown (temporal gap)
Explainability (weight visualization)
Failure case analysis
```

The system must remain fully functional and presentable with only the MUST HAVE column completed.
