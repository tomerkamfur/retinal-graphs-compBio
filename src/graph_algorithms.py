'''Graph algorithms utilities for retinal graph matrices.

This module provides:
- BFS shortest paths on unweighted adjacency matrix
- Dijkstra shortest paths on weighted adjacency matrix
- Endpoint-based connectivity metrics
'''

from __future__ import annotations

import argparse
import heapq
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_graph_matrices(graph_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    '''Load graph matrices from one image graph directory.'''
    a_path = graph_dir / "adjacency_unweighted.npy"
    w_path = graph_dir / "adjacency_weighted.npy"
    if not a_path.exists() or not w_path.exists():
        raise FileNotFoundError(f"Missing adjacency matrices in: {graph_dir}")
    a = np.load(a_path)
    w = np.load(w_path)
    return a, w


def load_nodes(graph_dir: Path) -> pd.DataFrame:
    '''Load nodes.csv if available.'''
    nodes_path = graph_dir / "nodes.csv"
    if not nodes_path.exists():
        return pd.DataFrame(columns=["id", "y", "x", "type"])
    return pd.read_csv(nodes_path)


def _neighbors_unweighted(a: np.ndarray, u: int) -> np.ndarray:
    return np.flatnonzero(a[u] > 0)


def _neighbors_weighted(w: np.ndarray, u: int) -> np.ndarray:
    return np.flatnonzero(w[u] > 0)


def bfs_shortest_paths(a: np.ndarray, source: int) -> np.ndarray:
    '''Shortest path lengths from source in hops (np.inf when unreachable).'''
    n = a.shape[0]
    dist = np.full(n, np.inf, dtype=float)
    dist[source] = 0.0
    q: deque[int] = deque([source])

    while q:
        u = q.popleft()
        for v in _neighbors_unweighted(a, u):
            if np.isinf(dist[v]):
                dist[v] = dist[u] + 1.0
                q.append(int(v))
    return dist


def dijkstra_shortest_paths(w: np.ndarray, source: int) -> np.ndarray:
    '''Shortest path lengths from source using non-negative edge weights.'''
    n = w.shape[0]
    dist = np.full(n, np.inf, dtype=float)
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist[u]:
            continue
        for v in _neighbors_weighted(w, u):
            weight = float(w[u, v])
            if weight < 0:
                raise ValueError("Dijkstra requires non-negative weights.")
            cand = d_u + weight
            if cand < dist[v]:
                dist[v] = cand
                heapq.heappush(pq, (cand, int(v)))
    return dist


def endpoint_node_ids(nodes_df: pd.DataFrame, num_nodes: int) -> np.ndarray:
    '''Return endpoint node ids; fallback to all nodes if unavailable.'''
    if "type" in nodes_df.columns and "id" in nodes_df.columns:
        endpoint_ids = nodes_df.loc[nodes_df["type"] == "endpoint", "id"].to_numpy()
        endpoint_ids = endpoint_ids.astype(int, copy=False)
        endpoint_ids = endpoint_ids[(endpoint_ids >= 0) & (endpoint_ids < num_nodes)]
        if endpoint_ids.size > 0:
            return np.unique(endpoint_ids)
    return np.arange(num_nodes, dtype=int)


def mean_endpoint_connectivity(
    a: np.ndarray,
    w: np.ndarray,
    endpoint_ids: Iterable[int],
) -> tuple[float, float]:
    '''Compute mean connectivity across endpoint nodes for BFS and Dijkstra.

    Connectivity of one endpoint = reachable other endpoints / total other endpoints.
    Returns (mean_bfs_connectivity, mean_dijkstra_connectivity) in [0, 1].
    '''
    endpoint_ids = np.array(list(endpoint_ids), dtype=int)
    m = endpoint_ids.size
    if m <= 1:
        return 0.0, 0.0

    bfs_scores = []
    dij_scores = []
    denom = float(m - 1)

    for src in endpoint_ids:
        bfs_dist = bfs_shortest_paths(a, int(src))
        dij_dist = dijkstra_shortest_paths(w, int(src))

        other = endpoint_ids[endpoint_ids != src]
        bfs_reachable = np.isfinite(bfs_dist[other]).sum()
        dij_reachable = np.isfinite(dij_dist[other]).sum()

        bfs_scores.append(float(bfs_reachable) / denom)
        dij_scores.append(float(dij_reachable) / denom)

    return float(np.mean(bfs_scores)), float(np.mean(dij_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BFS and Dijkstra on one graph directory.")
    parser.add_argument("--graph-dir", required=True, type=Path, help="Directory with adjacency matrices and nodes.csv")
    args = parser.parse_args()

    a, w = load_graph_matrices(args.graph_dir)
    nodes_df = load_nodes(args.graph_dir)
    endpoints = endpoint_node_ids(nodes_df, num_nodes=a.shape[0])
    bfs_conn, dij_conn = mean_endpoint_connectivity(a, w, endpoints)

    print(f"Graph: {args.graph_dir}")
    print(f"Nodes: {a.shape[0]}")
    print(f"Endpoint nodes used: {len(endpoints)}")
    print(f"Mean endpoint connectivity (BFS): {bfs_conn:.4f}")
    print(f"Mean endpoint connectivity (Dijkstra): {dij_conn:.4f}")


if __name__ == "__main__":
    main()
