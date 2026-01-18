"""Load and inspect the raw edges to see what paths are stored."""

import pickle
import sys
sys.path.insert(0, 'src')

# Re-run the graph building with debug output
from skeleton_to_graph import *
import numpy as np

# Load skeleton
skeleton = load_skeleton('outputs/skeleton.png')

# Detect nodes
node_pixels = detect_node_pixels(skeleton)
endpoints = node_pixels['endpoints']
junctions = node_pixels['junctions']

# Compress junctions
junction_centroids, junction_mask, labeled_junctions = compress_junction_clusters(junctions, skeleton)

# Build graph and get raw edges
edges_raw = trace_edges_from_nodes(skeleton, endpoints, junctions, junction_centroids, labeled_junctions)

# Find edges involving nodes 261 and 9
print("Looking for edges with nodes 261 or 9...")
for i, edge in enumerate(edges_raw):
    start_pix = edge['start']
    end_pix = edge['end']
    path = edge['path']
    
    print(f"\nEdge {i}: {start_pix} -> {end_pix}")
    print(f"  Path length (pixels): {len(path)}")
    print(f"  Path: {path}")
    
    # Calculate path_length
    path_len = 0.0
    for j in range(len(path) - 1):
        y1, x1 = path[j]
        y2, x2 = path[j + 1]
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        if (dy == 1 and dx == 0) or (dy == 0 and dx == 1):
            path_len += 1.0
        elif dy == 1 and dx == 1:
            path_len += np.sqrt(2)
        else:
            path_len += np.hypot(dy, dx)
    print(f"  Calculated path_length: {path_len:.4f}")
