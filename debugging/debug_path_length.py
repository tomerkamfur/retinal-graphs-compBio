'''Debug script to identify why path_length < euclidean in some edges.'''

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

# Find problematic edges where path_length < euclidean
print("Edges where path_length < euclidean:")
print("=" * 80)
problematic = []
for edge in edges:
    u, v = edge['u'], edge['v']
    path_len = edge['path_length']
    eucl = edge['euclidean']
    
    if path_len < eucl:
        y1, x1, type1 = nodes[u]
        y2, x2, type2 = nodes[v]
        
        # Check if nodes are on skeleton
        on_skel_u = skeleton[int(y1), int(x1)] if 0 <= int(y1) < skeleton.shape[0] and 0 <= int(x1) < skeleton.shape[1] else False
        on_skel_v = skeleton[int(y2), int(x2)] if 0 <= int(y2) < skeleton.shape[0] and 0 <= int(x2) < skeleton.shape[1] else False
        
        diff = eucl - path_len
        problematic.append({
            'u': u, 'v': v,
            'path_len': path_len,
            'eucl': eucl,
            'diff': diff,
            'u_type': type1,
            'v_type': type2,
            'u_on_skel': on_skel_u,
            'v_on_skel': on_skel_v,
        })

# Sort by difference
problematic.sort(key=lambda x: -x['diff'])

for edge in problematic[:20]:  # Show top 20
    print(f"Edge ({edge['u']},{edge['v']}): path={edge['path_len']:.4f}, eucl={edge['eucl']:.4f}, "
          f"diff={edge['diff']:.4f}, types=({edge['u_type']},{edge['v_type']}), "
          f"on_skel=({edge['u_on_skel']},{edge['v_on_skel']})")

print(f"\nTotal problematic edges: {len(problematic)} / {len(edges)}")
print(f"Percentage: {100 * len(problematic) / len(edges):.1f}%")

# Analyze node types in problematic edges
junction_start = sum(1 for e in problematic if e['u_type'] == 'junction')
junction_end = sum(1 for e in problematic if e['v_type'] == 'junction')
junction_both = sum(1 for e in problematic if e['u_type'] == 'junction' and e['v_type'] == 'junction')

print(f"\nProblematic edges by node type:")
print(f"  At least one junction: {sum(1 for e in problematic if e['u_type'] == 'junction' or e['v_type'] == 'junction')}")
print(f"  Junction as start: {junction_start}")
print(f"  Junction as end: {junction_end}")
print(f"  Both junctions: {junction_both}")
print(f"  Only endpoints: {sum(1 for e in problematic if e['u_type'] == 'endpoint' and e['v_type'] == 'endpoint')}")

# Analyze nodes off-skeleton
off_skel_u = sum(1 for e in problematic if not e['u_on_skel'])
off_skel_v = sum(1 for e in problematic if not e['v_on_skel'])
print(f"\nNode off-skeleton:")
print(f"  Start node off-skeleton: {off_skel_u}")
print(f"  End node off-skeleton: {off_skel_v}")
print(f"  At least one off-skeleton: {sum(1 for e in problematic if not e['u_on_skel'] or not e['v_on_skel'])}")
