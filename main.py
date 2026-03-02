"""End-to-end retinal graph pipeline.

Pipeline:
1. Find `data/messidor-2`
2. Convert all images to vessel masks
3. Skeletonize masks
4. Convert skeletons to graphs (nodes, edges, adjacency matrices)
5. Compute path/features statistics per image
6. Optionally create graph overlays with statistics
7. Build feature table
8. Build histograms by severity
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from skimage import io

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from preprocessing import SegmentationConfig, imread_rgb, save_mask_png, segment_vessel_mask
from preprocessing_maple_lib import (
    choose_threshold_auto,
    ensure_prob_2d,
    load_maple_tools,
    postprocess as maple_postprocess,
)
from skeletonize_vessels import compute_skeleton
from skeleton_to_graph import (
    build_adjacency_matrices,
    build_graph_from_skeleton,
    compress_junction_clusters,
    compute_edge_attributes,
    detect_node_pixels,
    plot_graph_overlay,
    save_results,
)
from graph_path_stats import (
    build_nx_graph,
    load_graph_data,
    pairwise_endpoint_paths,
    summarize_pairwise_results,
    tortuosity_edge_statistics,
)
from overlay_with_stats import annotate_image, compute_summary_from_results, edge_tortuosity_stats
from graph_feature_table import build_feature_table
from plot_feature_histograms import discover_feature_columns, plot_feature_by_severity


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_messidor_dir(data_root: Path) -> Path:
    messidor = data_root / "messidor-2"
    if not messidor.exists():
        raise FileNotFoundError(f"Could not find messidor-2 at: {messidor}")
    return messidor


def list_images(messidor_dir: Path) -> list[Path]:
    return [p for p in sorted(messidor_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def save_binary(path: Path, arr_bool: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, (arr_bool.astype(np.uint8) * 255))


def run_one_image(
    image_path: Path,
    out_dir: Path,
    cfg: SegmentationConfig,
    segmentation_mode: str,
    maple_ctx: dict | None,
    make_overlay: bool,
    overwrite: bool,
    verbose: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_id = image_path.stem
    result = {"image_name": image_id, "status": "ok", "error": ""}

    try:
        # 2) mask segmentation
        mask_path = out_dir / f"{image_id}_final_mask.png"
        if overwrite or not mask_path.exists():
            rgb = imread_rgb(image_path)
            if segmentation_mode == "maple":
                if maple_ctx is None:
                    raise RuntimeError("MAPLE context is not initialized.")

                args = SimpleNamespace(
                    resize=1500,
                    device=maple_ctx["device"],
                    model_threshold=0.5,
                    auto_threshold=True,
                    target_frac_min=0.10,
                    target_frac_max=0.25,
                    threshold=0.25,
                    min_area=80,
                    close_kernel=3,
                    open_kernel=0,
                )
                img = maple_ctx["circle_crop"](rgb, size=(args.resize, args.resize))
                prob = ensure_prob_2d(
                    maple_ctx["predict_fn"](
                        ensemble_models=maple_ctx["ensemble_models"],
                        images=img,
                        device=args.device,
                        threshold=args.model_threshold,
                    )
                )
                thr = choose_threshold_auto(prob, args.target_frac_min, args.target_frac_max)
                mask_u8 = maple_postprocess(
                    prob=prob,
                    threshold=thr,
                    min_area=args.min_area,
                    close_kernel=args.close_kernel,
                    open_kernel=args.open_kernel,
                )
                mask = mask_u8 > 0
                save_mask_png(mask, mask_path)
                if verbose:
                    frac = float(np.mean(mask))
                    print(f"[{image_id}] MAPLE mask saved (thr={thr:.3f}, coverage={frac:.4f})")
            else:
                mask, thr = segment_vessel_mask(rgb, cfg)
                save_mask_png(mask, mask_path)
                if verbose:
                    print(f"[{image_id}] regular mask saved ({thr['final_mask_fraction']:.4f} coverage)")
        else:
            mask = io.imread(mask_path) > 0
            rgb = imread_rgb(image_path)

        # 3) skeletonization
        skeleton = compute_skeleton(mask.astype(bool))
        skeleton_path = out_dir / "skeleton.png"
        save_binary(skeleton_path, skeleton)

        # 4) graph extraction
        node_pixels = detect_node_pixels(skeleton)
        endpoints = node_pixels["endpoints"]
        junctions = node_pixels["junctions"]
        junction_centroids, _, labeled_junctions = compress_junction_clusters(junctions, skeleton)
        nodes, edges_raw, pixel_to_node = build_graph_from_skeleton(
            skeleton, junctions, endpoints, junction_centroids, labeled_junctions
        )
        edges = compute_edge_attributes(edges_raw, nodes, pixel_to_node, labeled_junctions, junction_centroids)
        a, w = build_adjacency_matrices(len(nodes), edges, weight_attr="path_length")
        save_results(str(out_dir), nodes, edges, a, w, weight_attr="path_length")

        if make_overlay:
            overlay_path = out_dir / "graph_overlay.png"
            plot_graph_overlay(skeleton, nodes, edges, output_path=str(overlay_path))

        # 5) path/features statistics per-image
        nodes_l, endpoints_l, node_coords, a_l, w_l, edge_attr_map = load_graph_data(str(out_dir))
        g = build_nx_graph(a_l, w_l, edge_attr_map=edge_attr_map)
        pairwise = pairwise_endpoint_paths(g, endpoints_l, node_coords, edge_attr_map, weight_attr="weight")
        summarize_pairwise_results(pairwise, outdir=str(out_dir))
        edge_stats = tortuosity_edge_statistics(edge_attr_map)
        with open(out_dir / "tortuosity_edge_stats.txt", "w", encoding="utf-8") as f:
            f.write(str(edge_stats))

        # 6) optional overlay with stats
        if make_overlay:
            overlay_in = out_dir / "graph_overlay.png"
            overlay_out = out_dir / "graph_overlay_stats.png"
            summary = compute_summary_from_results(pairwise)
            edge_s = edge_tortuosity_stats(str(out_dir / "edges.csv"))
            metadata = {}
            meta_path = out_dir / "metadata.txt"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    for line in f:
                        if ":" in line:
                            k, v = line.strip().split(":", 1)
                            metadata[k.strip()] = v.strip()
            annotate_image(str(overlay_in), str(overlay_out), summary, edge_s, metadata)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def create_histograms(table_csv: Path, output_dir: Path, bins: int) -> None:
    import pandas as pd

    df = pd.read_csv(table_csv)
    if "severity_score" not in df.columns:
        raise ValueError("Feature table missing severity_score column.")
    df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce")
    df = df.dropna(subset=["severity_score"])

    severities = sorted(df["severity_score"].unique().tolist())
    features = discover_feature_columns(df, selected=None)
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        out = output_dir / f"hist_{feature}.png"
        plot_feature_by_severity(df, feature, severities, bins, out)


def write_run_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete retinal graph pipeline from messidor-2.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root data directory containing messidor-2")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/picturese_for_graphs"),
        help="Per-image output root directory",
    )
    parser.add_argument("--labels-csv", type=Path, default=Path("data/messidor_data.csv"))
    parser.add_argument("--feature-table-output", type=Path, default=Path("results/graph_feature_table.csv"))
    parser.add_argument("--hist-output-dir", type=Path, default=Path("results/feature_histograms"))
    parser.add_argument("--bins", type=int, default=10, help="Histogram bins")
    parser.add_argument("--overlay", action="store_true", help="Create graph overlays and overlay-with-stats images")
    parser.add_argument(
        "--segmentation",
        choices=["maple", "regular"],
        default="maple",
        help="Segmentation backend. Default is maple; use regular for the classic pipeline.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for MAPLE model: auto/cpu/cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")
    parser.add_argument(
        "--include-existing-graphs",
        action="store_true",
        help="If set, feature table/histograms include all graph folders in output-root. "
        "Default behavior includes only images processed successfully in the current run.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mask file")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    messidor_dir = find_messidor_dir(args.data_root)
    images = list_images(messidor_dir)
    if not images:
        print(f"No images found in {messidor_dir}")
        return 1

    if args.limit and args.limit > 0:
        images = images[: args.limit]

    cfg = SegmentationConfig(use_auto_thresholds=True)
    maple_ctx = None
    if args.segmentation == "maple":
        load_ensemble, predict_fn, circle_crop = load_maple_tools()
        device = args.device
        if device == "auto":
            try:
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        ensemble_models = load_ensemble(device=device)
        maple_ctx = {
            "predict_fn": predict_fn,
            "circle_crop": circle_crop,
            "ensemble_models": ensemble_models,
            "device": device,
        }

    print(f"Found {len(images)} images in {messidor_dir}")
    print(f"Output root: {args.output_root}")
    print(f"Segmentation mode: {args.segmentation}")

    rows = []
    for i, image_path in enumerate(images, start=1):
        out_dir = args.output_root / image_path.stem
        if args.verbose:
            print(f"\n[{i}/{len(images)}] Processing {image_path.name}")
        row = run_one_image(
            image_path=image_path,
            out_dir=out_dir,
            cfg=cfg,
            segmentation_mode=args.segmentation,
            maple_ctx=maple_ctx,
            make_overlay=args.overlay,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        rows.append(row)
        if row["status"] == "failed":
            print(f"[FAILED] {image_path.name}: {row['error']}")

    run_summary = Path("results") / "pipeline_run_summary.csv"
    write_run_summary(run_summary, rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nPer-image pipeline done. Success={ok}/{len(rows)}")
    print(f"Run summary: {run_summary}")

    # 7) feature table
    df = build_feature_table(args.output_root, args.labels_csv)
    if not args.include_existing_graphs:
        current_success = {r["image_name"] for r in rows if r["status"] == "ok"}
        df = df[df["image_name"].isin(current_success)].copy()
        df = df.sort_values("image_name").reset_index(drop=True)

    args.feature_table_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.feature_table_output, index=False)
    print(f"Saved feature table: {args.feature_table_output} (rows={len(df)})")

    # 8) histograms
    create_histograms(args.feature_table_output, args.hist_output_dir, bins=args.bins)
    print(f"Saved histograms to: {args.hist_output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
