"""Detailed trace of a specific problematic edge."""

import numpy as np
from scipy import ndimage
from skimage import io
import csv

def load_skeleton(image_path):
    """Load skeleton image and binarize."""
    img = io.imread(image_path)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    if img.dtype != bool:
        img = img.astype(float) / 255.0
    skeleton = img > 0.5
    return skeleton

# Load data
skeleton = load_skeleton('outputs/skeleton.png')

# Load edges
edges_by_id = {}
with open('graph_output/edges.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        u = int(row['u'])
        v = int(row['v'])
        path_len = float(row['path_length'])
        eucl = float(row['euclidean'])
        edges_by_id[(u, v)] = {'path_len': path_len, 'eucl': eucl}
        edges_by_id[(v, u)] = {'path_len': path_len, 'eucl': eucl}

# Find the worst problematic edge
worst_edge = (261, 9)
print(f"Edge {worst_edge}:")
print(f"  Path length: {edges_by_id[worst_edge]['path_len']}")
print(f"  Euclidean: {edges_by_id[worst_edge]['eucl']}")
print(f"  Difference: {edges_by_id[worst_edge]['eucl'] - edges_by_id[worst_edge]['path_len']}")

# Load nodes
nodes_by_id = {}
with open('graph_output/nodes.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        node_id = int(row['id'])
        y = float(row['y'])
        x = float(row['x'])
        node_type = row['type']
        nodes_by_id[node_id] = {'y': y, 'x': x, 'type': node_type}

u, v = worst_edge
y_u, x_u = nodes_by_id[u]['y'], nodes_by_id[u]['x']
y_v, x_v = nodes_by_id[v]['y'], nodes_by_id[v]['x']

print(f"\nNode {u} ({nodes_by_id[u]['type']}): ({y_u:.1f}, {x_u:.1f})")
print(f"Node {v} ({nodes_by_id[v]['type']}): ({y_v:.1f}, {x_v:.1f})")

# Check if nodes are on skeleton
on_skel_u = skeleton[int(y_u), int(x_u)]
on_skel_v = skeleton[int(y_v), int(x_v)]
print(f"\nOn skeleton: u={on_skel_u}, v={on_skel_v}")

# Try to manually trace from u to v
print(f"\nTrying to manually find shortest path on skeleton from ({int(y_u)}, {int(x_u)}) to ({int(y_v)}, {int(x_v)})...")

# BFS
from collections import deque

def get_8neighbors(y, x, shape):
    """Get 8-connected neighbors."""
    height, width = shape
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                neighbors.append((ny, nx))
    return neighbors

start = (int(y_u), int(x_u))
goal = (int(y_v), int(x_v))

queue = deque([(start, [start])])
visited = {start}
found = False

while queue and not found:
    (cy, cx), path = queue.popleft()
    
    if (cy, cx) == goal:
        print(f"Found path with {len(path)} pixels:")
        print(f"  Path: {path[:5]}... (showing first 5)")
        
        # Calculate path_length
        path_len = 0.0
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            dy = abs(y2 - y1)
            dx = abs(x2 - x1)
            if (dy == 1 and dx == 0) or (dy == 0 and dx == 1):
                path_len += 1.0
            elif dy == 1 and dx == 1:
                path_len += np.sqrt(2)
            else:
                path_len += np.hypot(dy, dx)
        
        print(f"  Calculated path_length: {path_len:.4f}")
        print(f"  Expected (from CSV): 1.0000")
        print(f"  Difference: {1.0 - path_len:.4f}")
        found = True
    
    for ny, nx in get_8neighbors(cy, cx, skeleton.shape):
        if (ny, nx) not in visited and skeleton[ny, nx]:
            visited.add((ny, nx))
            queue.append(((ny, nx), path + [(ny, nx)]))

if not found:
    print("No path found on skeleton!")
