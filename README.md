# Graph-Based Analysis of Retinal Blood Vessel Networks Across Diabetic Retinopathy Severity Levels

## Research Question
Do graph-based structural features extracted from retinal vessel networks (for example tortuosity, edge length, and connectivity) differ across diabetic retinopathy (DR) severity grades?

## Current End-to-End Pipeline
The project now supports a full pipeline from raw Messidor-2 fundus images to per-image graph features and severity-level histograms.

1. Vessel segmentation (raw image -> binary mask)
2. Skeletonization (mask -> 1-pixel centerline)
3. Graph extraction (nodes, edges, adjacency matrices)
4. Shortest-path statistics (endpoint pairs)
5. Feature aggregation table (per image + severity)
6. Histograms per feature by severity

## Main Entry Point
Run everything with:

```bash
python main.py
```

By default, `main.py` uses MAPLE-based segmentation.

Useful options:

```bash
# include graph overlay images
python main.py --overlay

# test on first 5 images
python main.py --limit 5 --verbose

# use the regular (non-MAPLE) segmentation instead
python main.py --segmentation regular
```

## Inputs
- `data/messidor-2/` : fundus images
- `data/messidor_data.csv` : labels with severity (`diagnosis`)

## Outputs

### Per-image outputs
Saved under:
- `data/picturese_for_graphs/<image_id>/`

Typical files:
- `<image_id>_final_mask.png`
- `skeleton.png`
- `nodes.csv`
- `edges.csv`
- `adjacency_unweighted.npy`
- `adjacency_weighted.npy`
- `endpoint_pair_paths.csv`
- `endpoint_pair_paths.npy`
- `metadata.txt`
- `tortuosity_path_stats.txt`
- `tortuosity_edge_stats.txt`
- `graph_overlay.png` (if `--overlay`)
- `graph_overlay_stats.png` (if `--overlay`)

### Global outputs
- `results/graph_feature_table.csv` : one row per image with graph features + severity
- `results/feature_histograms/` : histogram images per feature, split by severity
- `results/pipeline_run_summary.csv` : pipeline success/failure per image

## Core Scripts
- `main.py` : complete pipeline runner
- `src/preprocessing.py` : vessel mask extraction from retinal image
- `src/skeletonize_vessels.py` : skeleton generation
- `src/skeleton_to_graph.py` : graph building and adjacency matrix export
- `scripts/graph_path_stats.py` : pairwise endpoint shortest-path statistics
- `scripts/overlay_with_stats.py` : overlay annotation with summary stats
- `src/graph_algorithms.py` : BFS/Dijkstra helpers on saved matrices
- `src/graph_feature_table.py` : feature table creation + severity merge
- `src/plot_feature_histograms.py` : per-feature severity histograms

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```
