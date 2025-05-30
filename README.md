# ML Pipeline: Appearance Anomaly Detection

## System Requirements
This runs on an M4 macbook air with 16gb ram. I included some manual garbage collection to avoid OOM errors.

You must have `uv` installed to run this prototype.

To run API, run: `uv run -m uvicorn service:app --reload`

## Problem Statement

Detect contextual anomalies in human appearance over time within a stationary-camera video feed. The goal is not just identity continuity, but the detection of persistent visual changes (e.g., carrying a bag, showing an object) that may indicate anomalous or significant behavior.

This system prioritizes:

- Identity cluster stability
- Sensitivity to appearance drift
- Heuristic tolerances for YOLO failure modes (e.g., merged detections)

## High-Level Strategy

| Task | Current Method |
|------|----------------|
| Object Detection | YOLOv11n (person class only) |
| Identity Tracking | KMeans on OSNet embeddings, plus heuristics for intra-video identity |
| Scene Segmentation | ruptures.Pelt on binary activity signals per cluster |
| Appearance Drift Detection | Energy distance between PCA-reduced CLIP embedding distributions |
| Backend | DuckDB for ETL and modular pipeline state |
| Cross-video ReID | Stratified PCA + Spectral Clustering as a proxy for gait-based modeling |

### Technical Implementation
- **Device optimization**: Automatic MPS/CPU detection for Apple Silicon
- **Memory management**: Explicit model deletion after inference to prevent memory leaks
- **Batch processing**: Configurable batch sizes (OSNet: 32, CLIP: 32)
- **PCA dimensionality**: Consistent 50D reduction across OSNet and CLIP embeddings

> **Note:** No DeepSORT is used — clustering is naive but wrapped in a heuristic filter. The long-term goal is graph-aware constraint clustering, but current methods are tuned for MVP-level prototype fidelity.

## Key Pipeline Modules

### 1. Object Detection

**Model:** YOLOv11n

**Config:**
```python
YOLO_PARAMETERS = {
    "classes": [0],  # person
    "iou": 0.4,
    "conf": 0.5
}
```

- Runs inference frame-by-frame on input videos
- Detections are filtered for person class, then cropped + stored
- Outputs feed directly into identity clustering

### 2. Irregular Frame Detection

**Purpose:** Identify frames with unexpected detection counts (e.g., YOLO merging people when close together)

- Uses ruptures.Pelt on the `n_objects_detected` time series
- Flagged frames influence filtering and anomaly sampling

### 3. Identity Clustering (Intra-Video)

**Method:**
- OSNet embeddings → PCA (50D) → KMeans(n=2)

**Cleaned via:**
- Irregular frame mask
- Confidence quantile thresholds (bottom 5th percentile excluded)
- Multi-object heuristic: frames where >1 object maps to same cluster flagged as `is_bad_frame`

**Rationale:**
Designed to tolerate repeated trajectories with high visual similarity — e.g., the adult male repeatedly walking the same path.

**Long-term:** Replace with graph-constrained clustering or temporal affinity networks.

### 4. Scene Segmentation

**Method:**

For each identity cluster:
1. Create binary frame signal (active/inactive)
2. Apply `binary_closing` + `binary_opening` to smooth
3. Use `ruptures.Pelt` with l2 model to detect persistent appearance episodes

**Output:** Scene buckets per identity with `[start_frame, end_frame]` metadata

**Filtering constraints:**
- Minimum scene length: 150 frames
- Morphological smoothing: 10-frame kernels for opening/closing operations
- Change point penalty: 20 (ruptures.Pelt parameter)

## Appearance Anomaly Detection

### 5. Appearance Drift (Energy Distance)

- CLIP embeddings → PCA (50D)
- For each cluster, compare successive scene buckets using energy distance under cosine metric
- **Energy distance threshold**: 0.48 (empirically tuned)
- **Minimum bucket size**: 10 frames required for analysis
- **Stratified sampling**: Top 100 frames by normalized inter-arrival time (IAT) per scene bucket

```
energy_distance_cosine(A, B) = 2 * E[cosine(A, B)] - E[cosine(A, A)] - E[cosine(B, B)]
```

**Interpretation:** High energy = consistent contextual shift, e.g.:
- Change in carried object
- Different lighting or occlusion
- Person interacting with someone else

## Sampling Strategy

- Within each scene, frames are stratified by normalized inter-arrival time (IAT)
- Skews sampling toward sparser frames, which correlate with YOLO detection failure
- These often correspond to moments of interaction (e.g., when man shows the woman an object)

**Heuristic payoff:** Sampling sparse detection moments captures behavioral salience without relying on explicit object recognition or intent modeling. In other words, the man is more likely to have sparser captured frames when interacting with the woman due to the low IoU tolerance I set, which also happen to be the times where he's not occluding the object he's carrying (this happens in the second video when he carries the silver briefcase on the opposite side). This is a massive assumption which is tolerable for this MVP's scope but the sampling methodology would be revisited in the future.

The energy distance formula can be loosely understood as a nonparametric form of KL divergence. This is similar to another technique for capturing semantic drift, where jensen-shannon divergence is captured between distributions of embeddings at two points in time (the distribution can either be calculated as a discrete distribution of clusters – affinity propagation is used for this frequently – or a gaussian KDE).

## Database Schema & ETL Pipeline

The system uses DuckDB with a relational schema designed for modular pipeline processing:

**Core relationships:**
- `video` → `frame` → `detection` (1:many hierarchical structure)  
- `intravideo_object_ids` links detections to cluster assignments
- `scene` segments store temporal boundaries per cluster
- `crossvideo_reid` maintains global identity mappings across videos

**ETL pattern:** Each pipeline module reads from upstream tables, processes data in-memory, then writes results back to DuckDB for the next stage.

## Cross-Video ReID (Heuristic Only)

### 6. Spectral Clustering

- Sampled tracklet clusters (every 5th frame)
- OSNet embeddings → PCA (50D) → SpectralClustering (n=2)
- Used as a gait-free proxy to validate that the same person across videos is assigned consistent global identity
- **Confidence scoring**: Based on inclusion score (intersection_size / ground_truth_size)
- **Single video fallback**: Identity mapping reduces to pass-through when only one video present

**Note:** Real implementation would require gait modeling, which is out-of-scope for MVP.

## Evaluation

### ✅ Sanity Checks
- Exactly 2 stable clusters: 1 adult male, 1 young woman
- The woman should produce 0 anomalies throughout

### 🚨 Expected Anomalies
Man's cluster should spike in energy distance every time he changes appearance (i.e., is carrying a different object)

### 🔍 Anomaly Record Structure
Each anomaly record includes:
- `video_id`
- `cluster_id`
- `scene_bucket_index`
- `energy_distance_score`
- `is_anomaly`: bool

## Future Work

**Immediate improvements:**
- Replace KMeans with graph-constrained or temporal-aware clustering
- Add comprehensive error handling and validation throughout pipeline
- Make hard-coded parameters (cluster count, thresholds) configurable

**Research directions:**
- Integrate gait or pose-based re-identification
- Shift anomaly detection from heuristic to learned scoring (e.g., VAE, one-class SVM)
- Unify tracklets, scene segments, and global ReID into a graph-based framework

## Known Limitations

**System constraints:**
- **Hard-coded cluster count**: Pipeline assumes exactly 2 identities across all videos
- **Memory scaling**: Batch processing helps but large videos may still cause OOM issues
- **Minimal error handling**: Limited graceful degradation for edge cases (empty crops, failed embeddings)

**Algorithmic limitations:**
- **Temporal consistency**: No explicit trajectory smoothing or temporal constraints in clustering
- **YOLO failure modes**: Merged detections handled heuristically rather than systematically
- **Cross-video assumptions**: Fixed n_clusters=2 in SpectralClustering regardless of actual identity count
