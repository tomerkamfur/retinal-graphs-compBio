'''
Build a per-image graph feature table for downstream analysis.

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
- mean_branching_degree
'''

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
    '''Find directories that look like saved graph outputs.'''
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
    '''Load mapping image_id -> diagnosis severity from Messidor CSV.'''
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

def safe_percentile(series: pd.Series, q: float, default: float = 0.0) -> float:
    if series.empty:
        return default
    return float(np.percentile(series.to_numpy(dtype=float), q))

def safe_weighted_mean(values: pd.Series, weights: pd.Series, default: float = 0.0) -> float:
    if values.empty or weights.empty:
        return default
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return default
    return float(np.average(v[mask], weights=w[mask]))


def branching_degree_features(a: np.ndarray, nodes_df: pd.DataFrame) -> tuple[float, int]:
    '''
    Return (mean branching degree, branching node count).

    Branching nodes are junction nodes from nodes.csv. Degree is computed from
    the unweighted adjacency matrix row sums.
    '''
    if nodes_df.empty or "id" not in nodes_df.columns or "type" not in nodes_df.columns:
        return 0.0, 0

    junction_ids = nodes_df.loc[nodes_df["type"] == "junction", "id"].to_numpy(dtype=int, copy=False)
    junction_ids = junction_ids[(junction_ids >= 0) & (junction_ids < a.shape[0])]
    if junction_ids.size == 0:
        return 0.0, 0

    deg = np.sum(a[junction_ids] > 0, axis=1).astype(float)
    return float(np.mean(deg)), int(junction_ids.size)


def compute_graph_features(graph_dir: Path, severity_map: dict[str, float]) -> dict[str, float | str]:
    a, w = load_graph_matrices(graph_dir)
    nodes_df = load_nodes(graph_dir)
    edges_df = pd.read_csv(graph_dir / "edges.csv")
    pairwise_path_csv = graph_dir / "endpoint_pair_paths.csv"
    pairwise_df = pd.read_csv(pairwise_path_csv) if pairwise_path_csv.exists() else pd.DataFrame()

    endpoints = endpoint_node_ids(nodes_df, num_nodes=a.shape[0])
    bfs_conn, dij_conn = mean_endpoint_connectivity(a, w, endpoints)
    mean_branching_degree, branching_nodes_count = branching_degree_features(a, nodes_df)

    tort_series = edges_df["tortuosity"] if "tortuosity" in edges_df.columns else pd.Series(dtype=float)
    length_series = edges_df["path_length"] if "path_length" in edges_df.columns else pd.Series(dtype=float)
    long_edges_mask = length_series > 20 if not length_series.empty else pd.Series(dtype=bool)
    short_edges_mask = length_series < 10 if not length_series.empty else pd.Series(dtype=bool)
    long_edge_tort = tort_series[long_edges_mask] if not tort_series.empty and not length_series.empty else pd.Series(dtype=float)
    short_edges_count = int(np.sum(short_edges_mask)) if not length_series.empty else 0

    if "tortuosity" in pairwise_df.columns:
        path_tort = pd.to_numeric(pairwise_df["tortuosity"], errors="coerce")
        path_tort = path_tort[np.isfinite(path_tort)]
        mean_path_tortuosity = safe_mean(path_tort)
    else:
        mean_path_tortuosity = np.nan

    image_name = graph_dir.name
    image_id = normalize_image_id(image_name)

    row = {
        "image_name": image_name,
        "severity_score": severity_map.get(image_id, np.nan),
        "num_nodes": int(a.shape[0]),
        "num_edges": int(len(edges_df)),
        "mean_tortuosity": safe_mean(tort_series) if not tort_series.empty else np.nan,
        "mean_edge_length": safe_mean(edges_df["path_length"]) if "path_length" in edges_df.columns else np.nan,
        "length_weighted_mean_tortuosity": safe_weighted_mean(tort_series, length_series) if not tort_series.empty else np.nan,
        "mean_tortuosity_long_edges": safe_mean(long_edge_tort) if not long_edge_tort.empty else np.nan,
        "p90_tortuosity": safe_percentile(tort_series, 90) if not tort_series.empty else np.nan,
        "mean_path_tortuosity": mean_path_tortuosity,
        "short_edges_count": short_edges_count,
        "mean_connectivity_bfs": bfs_conn,
        "mean_connectivity_dijkstra": dij_conn,
        "mean_branching_degree": mean_branching_degree,
        "branching_nodes_count": branching_nodes_count,
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
