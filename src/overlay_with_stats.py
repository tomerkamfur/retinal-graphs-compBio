'''
Annotate graph overlay image with summary statistics.

Loads:
 - overlay image (default: graph_output/graph_overlay.png)
 - pairwise path results (default: stats_output/endpoint_pair_paths.npy)
 - metadata (graph_output/metadata.txt) and edge tortuosity (graph_output/edges.csv)

Saves annotated image (default: graph_output/graph_overlay_stats.png)
'''
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv


def load_pairwise_results(npy_path):
    if not os.path.exists(npy_path):
        return None
    data = np.load(npy_path, allow_pickle=True)
    # data is a list of dicts
    return list(data)


def compute_summary_from_results(results):
    if not results:
        return None
    unw = [r['unweighted_hops'] for r in results if np.isfinite(r['unweighted_hops'])]
    w = [r['weighted_cost'] for r in results if np.isfinite(r['weighted_cost'])]
    t = [r['tortuosity'] for r in results if np.isfinite(r['tortuosity'])]
    def stats(arr):
        a = np.array(arr)
        return {
            'count': int(a.size),
            'mean': float(np.mean(a)),
            'median': float(np.median(a)),
            'min': float(np.min(a)),
            'max': float(np.max(a)),
            'std': float(np.std(a))
        }
    return {'unweighted': stats(unw) if unw else None,
            'weighted': stats(w) if w else None,
            'tortuosity_paths': stats(t) if t else None}


def load_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        return {}
    md = {}
    with open(metadata_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.strip().split(':', 1)
                md[k.strip()] = v.strip()
    return md


def edge_tortuosity_stats(edges_csv):
    if not os.path.exists(edges_csv):
        return None
    torts = []
    with open(edges_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row.get('tortuosity', 'nan'))
            except Exception:
                continue
            if np.isfinite(t):
                torts.append(t)
    if not torts:
        return None
    a = np.array(torts)
    return {'count': int(a.size), 'mean': float(a.mean()), 'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max()), 'std': float(a.std())}


def annotate_image(overlay_path, out_path, summary, edge_stats, metadata):
    img = plt.imread(overlay_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.axis('off')

    # Build text lines
    lines = []
    if metadata:
        nodes = metadata.get('Nodes') or metadata.get('Nodes', '')
        edges = metadata.get('Edges') or metadata.get('Edges', '')
        lines.append(f"Nodes: {nodes}")
        lines.append(f"Edges: {edges}")
    if summary:
        p = summary.get('tortuosity_paths')
        if p:
            lines.append('Path tortuosity:')
            lines.append(f"  mean {p['mean']:.3f}  med {p['median']:.3f}  sd {p['std']:.3f}")
    if edge_stats:
        lines.append('Edge tortuosity:')
        lines.append(f"  mean {edge_stats['mean']:.3f}  med {edge_stats['median']:.3f}  sd {edge_stats['std']:.3f}")

    text = '\n'.join(lines)
    bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=10, va='bottom', ha='left', bbox=bbox_props)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlay', default='graph_output/graph_overlay.png')
    parser.add_argument('--pairwise', default='stats_output/endpoint_pair_paths.npy')
    parser.add_argument('--edges', default='graph_output/edges.csv')
    parser.add_argument('--metadata', default='graph_output/metadata.txt')
    parser.add_argument('--out', default='graph_output/graph_overlay_stats.png')
    args = parser.parse_args()

    results = load_pairwise_results(args.pairwise)
    summary = compute_summary_from_results(results)
    metadata = load_metadata(args.metadata)
    edge_stats = edge_tortuosity_stats(args.edges)

    out_path = annotate_image(args.overlay, args.out, summary, edge_stats, metadata)
    print('Saved annotated overlay:', out_path)

if __name__ == '__main__':
    main()
