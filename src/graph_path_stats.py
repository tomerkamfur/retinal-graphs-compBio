'''Compute shortest-path statistics (unweighted and weighted) between endpoint nodes.

Reads adjacency matrices and node/edge CSVs from a graph output directory (NPY + CSV),
then computes pairwise shortest paths between endpoints (undirected, consider each pair once).
Also computes tortuosity statistics for edge set and for computed paths.

Usage:
    python scripts/graph_path_stats.py --graphdir graph_output --outdir stats_output --weight weight

Outputs saved in `outdir`:
 - endpoint_pair_paths.csv  (u,v,unweighted_hops,weighted_cost,path_length_sum,euclidean,tortuosity)
 - path_stats.npy (numpy structured summary arrays)
 - tortuosity_edge_stats.txt
 - tortuosity_path_stats.txt

Requirements: numpy, scikit-image, networkx, scipy
'''
import argparse
import os
import csv
import numpy as np
import networkx as nx
from collections import defaultdict


def load_graph_data(graphdir):
    '''Load nodes.csv, edges.csv, adjacency npy files from graphdir.

    Returns:
        nodes: list of (id, y, x, type)
        endpoints: list of endpoint node ids
        node_coords: dict node_id -> (y,x)
        A: unweighted adjacency numpy array
        W: weighted adjacency numpy array
        edge_attr_map: dict[(u,v)] -> {path_length, euclidean, tortuosity, weight}
    '''
    nodes_csv = os.path.join(graphdir, 'nodes.csv')
    edges_csv = os.path.join(graphdir, 'edges.csv')
    A_npy = os.path.join(graphdir, 'adjacency_unweighted.npy')
    W_npy = os.path.join(graphdir, 'adjacency_weighted.npy')

    if not os.path.exists(nodes_csv) or not os.path.exists(A_npy) or not os.path.exists(W_npy):
        raise FileNotFoundError('Required graph files missing in graphdir')

    # Load nodes
    nodes = []
    node_coords = {}
    endpoints = []
    with open(nodes_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row['id'])
            y = float(row['y'])
            x = float(row['x'])
            ntype = row['type']
            nodes.append((nid, y, x, ntype))
            node_coords[nid] = (y, x)
            if ntype == 'endpoint':
                endpoints.append(nid)

    A = np.load(A_npy)
    W = np.load(W_npy)

    # Load edge attributes if edges.csv exists
    edge_attr_map = {}
    if os.path.exists(edges_csv):
        with open(edges_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    u = int(row['u'])
                    v = int(row['v'])
                except Exception:
                    continue
                key = tuple(sorted((u, v)))
                attr = {
                    'path_length': float(row.get('path_length', 0)),
                    'euclidean': float(row.get('euclidean', 0)),
                    'tortuosity': float(row.get('tortuosity', 0)),
                    'weight': float(row.get('weight', 0))
                }
                edge_attr_map[key] = attr

    return nodes, endpoints, node_coords, A, W, edge_attr_map


def build_nx_graph(A, W, edge_attr_map=None):
    '''Build undirected NetworkX graph from adjacency matrices and attach attributes.

    Edge attributes attached from edge_attr_map when available.
    '''
    num_nodes = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for u in range(num_nodes):
        for v in range(u+1, num_nodes):
            if A[u, v]:
                weight = float(W[u, v])
                G.add_edge(u, v, weight=weight)
                if edge_attr_map is not None:
                    key = (u, v)
                    if key in edge_attr_map:
                        for k, val in edge_attr_map[key].items():
                            G[u][v][k] = val
    return G


def pairwise_endpoint_paths(G, endpoints, node_coords, edge_attr_map, weight_attr='weight'):
    '''Compute shortest paths between unordered endpoint pairs.

    Returns list of dicts with fields: u, v, unweighted_hops, weighted_cost, path_length_sum, euclidean, tortuosity
    '''
    endpoints_sorted = sorted(endpoints)
    n = len(endpoints_sorted)
    results = []

    # Precompute single-source shortest paths (unweighted hops) for each endpoint using BFS
    # and weighted distances using Dijkstra (with weight_attr)
    for i, u in enumerate(endpoints_sorted):
        # Unweighted distances and predecessors
        try:
            length_unweighted = nx.single_source_shortest_path_length(G, u)
        except Exception:
            length_unweighted = {}
        # Weighted distances and paths
        try:
            dist_weighted, path_weighted = nx.single_source_dijkstra(G, u, weight=weight_attr)
        except Exception:
            dist_weighted = {}
            path_weighted = {}

        for v in endpoints_sorted[i+1:]:
            unw = length_unweighted.get(v, np.inf)
            wcost = dist_weighted.get(v, np.inf)

            # path nodes for weighted shortest path
            path_nodes = path_weighted.get(v, None)

            # Compute summed path_length along edges (if edge_attr_map has path_length)
            path_length_sum = None
            tortuosity = None
            euclidean = None
            if path_nodes is not None and len(path_nodes) >= 2:
                # Sum path_length from edges (if available), else sum number of hops
                total_steps = 0.0
                total_step_count = 0
                for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                    key = tuple(sorted((a, b)))
                    if key in edge_attr_map:
                        total_steps += edge_attr_map[key].get('path_length', 1.0)
                    else:
                        total_steps += 1.0
                    total_step_count += 1
                path_length_sum = total_steps

                # Euclidean distance between endpoints using node_coords
                ya, xa = node_coords[u]
                yb, xb = node_coords[v]
                euclidean = float(np.hypot(yb - ya, xb - xa))

                tortuosity = float(path_length_sum / max(euclidean, 1e-6))
            else:
                # No path found between these endpoints
                path_length_sum = np.inf
                euclidean = np.hypot(node_coords[v][0] - node_coords[u][0], node_coords[v][1] - node_coords[u][1])
                tortuosity = np.inf

            results.append({
                'u': u,
                'v': v,
                'unweighted_hops': float(unw) if np.isfinite(unw) else np.inf,
                'weighted_cost': float(wcost) if np.isfinite(wcost) else np.inf,
                'path_length_sum': float(path_length_sum),
                'euclidean': float(euclidean),
                'tortuosity': float(tortuosity)
            })
    return results


def stats_from_values(values):
    arr = np.array([v for v in values if np.isfinite(v)])
    if arr.size == 0:
        return None
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'std': float(np.std(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75))
    }


def summarize_pairwise_results(results, outdir=None):
    '''Compute summary statistics for unweighted hops, weighted cost, and tortuosity across pairs.'''
    unweighted = [r['unweighted_hops'] for r in results]
    weighted = [r['weighted_cost'] for r in results]
    tortu = [r['tortuosity'] for r in results]

    s_unw = stats_from_values(unweighted)
    s_w = stats_from_values(weighted)
    s_t = stats_from_values(tortu)

    summary = {
        'unweighted_hops': s_unw,
        'weighted_cost': s_w,
        'tortuosity_paths': s_t,
        'total_pairs': len(results)
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # Save CSV of pairwise
        csv_path = os.path.join(outdir, 'endpoint_pair_paths.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['u', 'v', 'unweighted_hops', 'weighted_cost', 'path_length_sum', 'euclidean', 'tortuosity'])
            for r in results:
                writer.writerow([r['u'], r['v'], r['unweighted_hops'], r['weighted_cost'], r['path_length_sum'], r['euclidean'], r['tortuosity']])
        np.save(os.path.join(outdir, 'endpoint_pair_paths.npy'), results)

        # Save tortuosity summaries
        with open(os.path.join(outdir, 'tortuosity_path_stats.txt'), 'w') as f:
            f.write(str(s_t))
        print(f"Saved pairwise CSV and NPY to {outdir}")

    return summary


def tortuosity_edge_statistics(edge_attr_map):
    '''Return statistics for tortuosity values across edges.'''
    torts = [v['tortuosity'] for v in edge_attr_map.values() if np.isfinite(v.get('tortuosity', np.nan))]
    return stats_from_values(torts)


def main():
    parser = argparse.ArgumentParser(description='Compute shortest-path stats between endpoint nodes')
    parser.add_argument('--graphdir', required=True, help='Directory with adjacency npy and nodes.csv')
    parser.add_argument('--outdir', required=True, help='Directory to write results')
    parser.add_argument('--weight', choices=['weight', 'tortuosity', 'path_length', 'euclidean'], default='weight', help='Edge attribute to use as graph weight for Dijkstra')
    args = parser.parse_args()

    nodes, endpoints, node_coords, A, W, edge_attr_map = load_graph_data(args.graphdir)
    G = build_nx_graph(A, W, edge_attr_map=edge_attr_map)

    print(f"Loaded graph with {len(nodes)} nodes, {len(endpoints)} endpoints")

    results = pairwise_endpoint_paths(G, endpoints, node_coords, edge_attr_map, weight_attr=args.weight)

    summary = summarize_pairwise_results(results, outdir=args.outdir)
    edge_t_stats = tortuosity_edge_statistics(edge_attr_map)

    # Save summary files
    with open(os.path.join(args.outdir, 'summary.txt'), 'w') as f:
        f.write('Pairwise summary:\n')
        f.write(str(summary))
        f.write('\nEdge tortuosity summary:\n')
        f.write(str(edge_t_stats))

    print('Summary:')
    print(summary)
    print('\nEdge tortuosity stats:')
    print(edge_t_stats)


if __name__ == '__main__':
    main()
