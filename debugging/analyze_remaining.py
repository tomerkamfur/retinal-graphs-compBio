'''Detailed analysis of remaining problematic edges.'''

import numpy as np
from scipy import ndimage
from skimage import io
import csv

def load_skeleton(image_path):
    '''Load skeleton image and binarize.'''
    img = io.imread(image_path)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    if img.dtype != bool:
        img = img.astype(float) / 255.0
    skeleton = img > 0.5
    return skeleton

# Load data
skeleton = load_skeleton('outputs/skeleton.png')

# Load node coordinates
nodes = {}
with open('graph_output/nodes.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        node_id = int(row['id'])
        y = float(row['y'])
        x = float(row['x'])
        node_type = row['type']
        nodes[node_id] = (y, x, node_type)

# Load edges
edges = []
with open('graph_output/edges.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        u = int(row['u'])
        v = int(row['v'])
        path_length = float(row['path_length'])
        euclidean = float(row['euclidean'])
        edges.append({
            'u': u, 'v': v, 
            'path_length': path_length, 
            'euclidean': euclidean
        })

# Find problematic edges
print("Analysis of edges where path_length < euclidean:")
print("=" * 90)

for edge in edges:
    u, v = edge['u'], edge['v']
    path_len = edge['path_length']
    eucl = edge['euclidean']
    
    if path_len < eucl:
        y1, x1, type1 = nodes[u]
        y2, x2, type2 = nodes[v]
        
        # Check diagonal distance between nodes
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        
        # Expected minimum path considering only diagonals + orthogonal
        # Number of diagonal steps = min(dy, dx)
        # Remaining orthogonal steps = abs(dy - dx)
        diag_steps = min(dy, dx)
        orth_steps = abs(dy - dx)
        expected_min = diag_steps * np.sqrt(2) + orth_steps
        
        diff = eucl - path_len
        
        print(f"\nEdge ({u},{v}): path={path_len:.4f}, eucl={eucl:.4f}, diff={diff:.4f}")
        print(f"  Types: ({type1}, {type2})")
        print(f"  Node 1: ({y1:.1f}, {x1:.1f}), Node 2: ({y2:.1f}, {x2:.1f})")
        print(f"  Δy={dy:.1f}, Δx={dx:.1f}")
        print(f"  Expected min path (diagonal+ortho): {expected_min:.4f}")
        print(f"  Ratio path/eucl: {path_len/eucl:.4f}")
        
        # The problem: path should be >= euclidean by Euclidean geometry
        # But if it's not, possible causes:
        # 1. Path calculation error (shouldn't happen with our diagonal-aware code)
        # 2. The path doesn't actually exist (edge detection issue)
        # 3. The path found is a shortcut through the skeleton
