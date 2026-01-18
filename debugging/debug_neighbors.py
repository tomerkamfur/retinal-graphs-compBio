"""Debug the edge tracing algorithm."""

import numpy as np
from scipy import ndimage
from skimage import io

def load_skeleton(image_path):
    """Load skeleton image and binarize."""
    img = io.imread(image_path)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    if img.dtype != bool:
        img = img.astype(float) / 255.0
    skeleton = img > 0.5
    return skeleton

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

def count_neighbors(skeleton, y, x):
    """Count skeleton neighbors."""
    neighbors = get_8neighbors(y, x, skeleton.shape)
    return sum(1 for ny, nx in neighbors if skeleton[ny, nx])

# Load skeleton
skeleton = load_skeleton('outputs/skeleton.png')

# Test the two nodes
u_pos = (754, 202)  # Junction 261
v_pos = (752, 201)  # Endpoint 9

print(f"Start pos: {u_pos}, on_skel={skeleton[u_pos]}")
print(f"End pos: {v_pos}, on_skel={skeleton[v_pos]}")

print(f"\nNeighbors of start ({u_pos}):")
neighbors_start = get_8neighbors(u_pos[0], u_pos[1], skeleton.shape)
for ny, nx in neighbors_start:
    on_skel = skeleton[ny, nx]
    print(f"  ({ny}, {nx}): {on_skel}")

print(f"\nNeighbors of end ({v_pos}):")
neighbors_end = get_8neighbors(v_pos[0], v_pos[1], skeleton.shape)
for ny, nx in neighbors_end:
    on_skel = skeleton[ny, nx]
    print(f"  ({ny}, {nx}): {on_skel}")

# Check if they're actually connected
print(f"\nDirect neighbors shared: {set(neighbors_start) & set(neighbors_end)}")

# Count neighbors for node classification
print(f"\nNode classification (skeleton neighbors):")
print(f"  Start node {u_pos}: {count_neighbors(skeleton, u_pos[0], u_pos[1])} neighbors (junction>2, endpoint=1)")
print(f"  End node {v_pos}: {count_neighbors(skeleton, v_pos[0], v_pos[1])} neighbors")
