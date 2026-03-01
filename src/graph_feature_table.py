"""Build a per-image graph feature table for downstream analysis.

Expected per-image graph directory files:
- adjacency_unweighted.npy
- adjacency_weighted.npy
- nodes.csv
- edges.csv

Output table columns include:
- image_name
- severity_score
- num_nodes
- num_edges
- mean_tortuosity
- mean_edge_length
- mean_connectivity_bfs
- mean_connectivity_dijkstra
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from graph_algorithms import (
    endpoint_node_ids,
    load_graph_matrices,
    load_nodes,
    mean_endpoint_connectivity,
)


def discover_graph_dirs(root: Path) -> list[Path]:
    """Find directories that look like saved graph outputs."""
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        if (
            (p / "adjacency_unweighted.npy").exists()
            and (p / "adjacency_weighted.npy").exists()
            and (p / "edges.csv").exists()
        ):
            out.append(p)
    return out


def normalize_image_id(name: str) -> str:
    return Path(name).stem


def load_severity_map(labels_csv: Path) -> dict[str, float]:
    """Load mapping image_id -> diagnosis severity from Messidor CSV."""
    df = pd.read_csv(labels_csv)
    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        raise ValueError("labels CSV must contain columns: id_code, diagnosis")

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        image_id = normalize_image_id(str(row["id_code"]))
        out[image_id] = float(row["diagnosis"])
    return out


def safe_mean(series: pd.Series, default: float = 0.0) -> float:
    if series.empty:
        return default
    return float(series.mean())


def compute_graph_features(graph_dir: Path, severity_map: dict[str, float]) -> dict[str, float | str]:
    a, w = load_graph_matrices(graph_dir)
    nodes_df = load_nodes(graph_dir)
    edges_df = pd.read_csv(graph_dir / "edges.csv")

    endpoints = endpoint_node_ids(nodes_df, num_nodes=a.shape[0])
    bfs_conn, dij_conn = mean_endpoint_connectivity(a, w, endpoints)

    image_name = graph_dir.name
    image_id = normalize_image_id(image_name)

    row = {
        "image_name": image_name,
        "severity_score": severity_map.get(image_id, np.nan),
        "num_nodes": int(a.shape[0]),
        "num_edges": int(len(edges_df)),
        "mean_tortuosity": safe_mean(edges_df["tortuosity"]) if "tortuosity" in edges_df.columns else np.nan,
        "mean_edge_length": safe_mean(edges_df["path_length"]) if "path_length" in edges_df.columns else np.nan,
        "mean_connectivity_bfs": bfs_conn,
        "mean_connectivity_dijkstra": dij_conn,
    }
    return row


def build_feature_table(graphs_root: Path, labels_csv: Path) -> pd.DataFrame:
    severity_map = load_severity_map(labels_csv)
    graph_dirs = discover_graph_dirs(graphs_root)
    if not graph_dirs:
        raise FileNotFoundError(f"No graph directories found in: {graphs_root}")

    rows = []
    for graph_dir in graph_dirs:
        rows.append(compute_graph_features(graph_dir, severity_map))

    df = pd.DataFrame(rows)
    return df.sort_values("image_name").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate graph features into one table.")
    parser.add_argument("--graphs-root", type=Path, required=True, help="Root folder containing per-image graph folders")
    parser.add_argument("--labels-csv", type=Path, default=Path("data/messidor_data.csv"), help="Messidor labels CSV")
    parser.add_argument("--output-csv", type=Path, required=True, help="Where to write the feature table CSV")
    args = parser.parse_args()

    df = build_feature_table(args.graphs_root, args.labels_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved table: {args.output_csv}")
    print(f"Rows: {len(df)}")
    missing = int(df["severity_score"].isna().sum())
    print(f"Rows missing severity: {missing}")


if __name__ == "__main__":
    main()
