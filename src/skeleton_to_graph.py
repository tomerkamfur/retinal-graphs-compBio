"""Convert a 1-pixel-wide skeleton image to a graph representation with adjacency matrices.

Pipeline:
1) Load skeleton image and binarize (True = vessel).
2) Detect nodes:
   - endpoints: skeleton pixels with exactly 1 neighbor (8-connectivity)
   - junctions: skeleton pixels with >= 3 neighbors (8-connectivity)
   - compress junction clusters using connected components; use centroid as node location
3) Trace edges between nodes:
   - For each node, walk along skeleton to find neighboring nodes
   - Record pixel path for each segment
   - Prevent duplicate edges
4) Compute edge attributes:
   - path_length: number of steps (or sum of Euclidean distances)
   - euclidean: straight-line distance between node coordinates
   - tortuosity: path_length / max(euclidean, eps)
   - weight: tortuosity (or path_length, configurable)
5) Output:
   - Adjacency matrix (NxN unweighted 0/1)
   - Weighted adjacency matrix (NxN with edge weights)
   - Node list CSV (id, x, y, type)
   - Edge list CSV (u, v, path_length, euclidean, tortuosity, weight)
6) Quality control:
   - Print node/edge/component counts
   - Optionally plot overlay on skeleton

Usage:
    python src/skeleton_to_graph.py --image path/to/skeleton.png --outdir out_dir --weight tortuosity
    python src/skeleton_to_graph.py --image outputs/skeleton.png --outdir graph_out --weight path_length --plot
"""

import argparse
import os
import numpy as np
from scipy import ndimage
from skimage import io, img_as_bool
import csv


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_skeleton(image_path):
    """Load skeleton image and binarize.
    
    Args:
        image_path: Path to skeleton PNG (white on black).
    
    Returns:
        Boolean array where True = skeleton pixel.
    """
    img = io.imread(image_path)
    # Handle grayscale or RGB
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    # Normalize to [0, 1] if needed
    if img.dtype != bool:
        img = img.astype(float) / 255.0
    skeleton = img > 0.5
    return skeleton


def get_8neighbors(y, x, shape):
    """Get 8-connected neighbor coordinates (y, x) for position (y, x).
    
    Args:
        y, x: row and column indices
        shape: (height, width) of array
    
    Returns:
        List of (y, x) tuples for valid neighbors.
    """
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


def count_skeleton_neighbors(skeleton, y, x):
    """Count how many 8-connected neighbors are skeleton pixels.
    
    Args:
        skeleton: Boolean array
        y, x: Position to check
    
    Returns:
        Number of skeleton neighbors (0-8).
    """
    neighbors = get_8neighbors(y, x, skeleton.shape)
    count = sum(1 for ny, nx in neighbors if skeleton[ny, nx])
    return count


def detect_node_pixels(skeleton):
    """Detect node pixels: endpoints (1 neighbor) and junctions (>= 3 neighbors).
    
    Args:
        skeleton: Boolean array
    
    Returns:
        Dictionary with keys 'endpoints' and 'junctions', each a set of (y, x) tuples.
    """
    endpoints = set()
    junctions = set()
    
    ys, xs = np.where(skeleton)
    for y, x in zip(ys, xs):
        count = count_skeleton_neighbors(skeleton, y, x)
        if count == 1:
            endpoints.add((y, x))
        elif count >= 3:
            junctions.add((y, x))
    
    return {'endpoints': endpoints, 'junctions': junctions}


def compress_junction_clusters(junctions, skeleton):
    """Merge nearby junction pixels into single nodes using connected components.
    
    Each junction cluster is represented by its nearest-skeleton-pixel centroid to ensure
    all node coordinates lie exactly on the skeleton. This prevents geometric impossibilities
    where path_length < euclidean distance.
    
    Args:
        junctions: Set of (y, x) tuples for junction pixels
        skeleton: Boolean array (used for shape reference)
    
    Returns:
        centroids: Dict mapping label_id -> (y_c, x_c) as closest skeleton pixel to centroid
        junction_mask: Boolean mask of junction pixels
        labeled: Labeled connected component array
    """
    if not junctions:
        return {}, np.zeros_like(skeleton, dtype=bool), np.zeros_like(skeleton, dtype=int)
    
    # Create a mask of junction pixels
    junction_mask = np.zeros_like(skeleton, dtype=bool)
    for y, x in junctions:
        junction_mask[y, x] = True
    
    # Label connected components of junctions
    labeled, num_features = ndimage.label(junction_mask, structure=np.ones((3, 3), dtype=int))
    
    # For each component, compute centroid then snap to nearest skeleton pixel
    centroids = {}
    for label_id in range(1, num_features + 1):
        ys, xs = np.where(labeled == label_id)
        centroid_y = np.mean(ys)
        centroid_x = np.mean(xs)
        
        # Snap to nearest junction pixel (all junction pixels are on skeleton by definition)
        min_dist = float('inf')
        best_y, best_x = int(centroid_y), int(centroid_x)
        for y, x in zip(ys, xs):
            dist = (y - centroid_y)**2 + (x - centroid_x)**2
            if dist < min_dist:
                min_dist = dist
                best_y, best_x = y, x
        
        centroids[label_id] = (float(best_y), float(best_x))
    
    return centroids, junction_mask, labeled


def build_node_list(endpoints, junctions, junction_centroids, labeled_junctions):
    """Build a node list with (id, y, x, type) for all detected nodes.
    
    Args:
        endpoints: Set of (y, x) tuples
        junctions: Set of (y, x) tuples (original junction pixels)
        junction_centroids: Dict mapping label_id -> (y_c, x_c) snapped centroids
        labeled_junctions: Labeled array of junction clusters
    
    Returns:
        nodes: List of tuples (node_id, y, x, node_type).
        pixel_to_node: Mapping from (y, x) -> node_id for tracing purposes.
                      - Endpoints map to themselves
                      - Original junction pixels map to their snapped centroid's node
    """
    nodes = []
    node_id = 0
    pixel_to_node = {}  # Maps skeleton pixel to node_id
    
    # Add endpoints
    for y, x in endpoints:
        nodes.append((node_id, float(y), float(x), 'endpoint'))
        pixel_to_node[(y, x)] = node_id
        node_id += 1
    
    # Add junctions (using snapped centroids)
    # Build mapping: label_id -> node_id
    label_to_node_id = {}
    for label_id, (y_c, x_c) in junction_centroids.items():
        nodes.append((node_id, float(y_c), float(x_c), 'junction'))
        label_to_node_id[label_id] = node_id
        # Map the snapped pixel itself to the node
        pixel_to_node[(int(y_c), int(x_c))] = node_id
        node_id += 1
    
    # Also map ALL original junction pixels to their snapped node_id
    # This ensures path tracing finds them
    if labeled_junctions is not None:
        ys, xs = np.where(labeled_junctions > 0)
        for y, x in zip(ys, xs):
            label_id = labeled_junctions[y, x]
            if label_id in label_to_node_id:
                pixel_to_node[(y, x)] = label_to_node_id[label_id]
    
    return nodes, pixel_to_node


def trace_edges_from_nodes(skeleton, endpoints, junctions, junction_centroids, labeled_junctions):
    """Trace edges by walking from each node to neighboring nodes.
    
    Args:
        skeleton: Boolean array
        endpoints: Set of (y, x) endpoint pixels
        junctions: Set of (y, x) junction pixels (original, not used for node positions)
        junction_centroids: Dict mapping label_id to (y_c, x_c) snapped centroids
        labeled_junctions: Labeled array from connected components
    
    Returns:
        List of edges, where each edge is:
        {
            'start': start_pixel,
            'end': end_pixel,
            'path': [(y1, x1), (y2, x2), ...]
        }
    """
    # Build all_nodes using SNAPPED junction centroids, not original junction pixels
    all_nodes = endpoints.copy()
    # Add snapped junction centroids (as integer tuples for node positions)
    for label_id, (y_c, x_c) in junction_centroids.items():
        all_nodes.add((int(y_c), int(x_c)))
    
    visited_steps = set()  # Track visited (from_pixel, to_pixel) directed steps
    edges = []
    
    for start_pixel in all_nodes:
        # Get skeleton neighbors of this node
        start_y, start_x = start_pixel
        neighbors = get_8neighbors(start_y, start_x, skeleton.shape)
        skeleton_neighbors = [(ny, nx) for ny, nx in neighbors if skeleton[ny, nx]]
        
        # For each neighbor, try to trace a path
        for next_y, next_x in skeleton_neighbors:
            step = (start_pixel, (next_y, next_x))
            if step in visited_steps:
                continue  # Already processed this direction
            
            visited_steps.add(step)
            
            # Trace from (next_y, next_x) back to find the next node
            path, end_pixel = trace_path(skeleton, start_pixel, (next_y, next_x), 
                                        all_nodes, junctions, labeled_junctions)
            
            if end_pixel is not None and end_pixel != start_pixel:
                # Valid edge found
                edges.append({
                    'start': start_pixel,
                    'end': end_pixel,
                    'path': path
                })
                # Mark reverse direction as visited
                reverse_step = (end_pixel, path[-1])  # From end node's perspective
                if len(path) > 1:
                    reverse_step = (end_pixel, path[-2])
                # Avoid double marking; just rely on visited_steps at start
    
    return edges


def trace_path(skeleton, start_pixel, current_pixel, all_nodes, junctions, labeled_junctions):
    """Trace a path from current_pixel until reaching another node.
    
    Args:
        skeleton: Boolean array
        start_pixel: (y, x) where we started walking from
        current_pixel: (y, x) current position
        all_nodes: Set of all node pixels
        junctions: Set of junction pixels
        labeled_junctions: Labeled array for junction clustering
    
    Returns:
        (path_pixels, end_node_pixel) or (None, None) if path terminates
        where path_pixels = [(y1, x1), (y2, x2), ...]
    """
    path = [current_pixel]
    prev_pixel = start_pixel
    
    max_steps = 10000  # Prevent infinite loops
    steps = 0
    
    while steps < max_steps:
        steps += 1
        curr_y, curr_x = current_pixel
        
        # Check if we've reached a node (other than start)
        if current_pixel in all_nodes and current_pixel != start_pixel:
            # Check if this is a compressed junction (same label as start's label if applicable)
            return path, current_pixel
        
        # Get skeleton neighbors
        neighbors = get_8neighbors(curr_y, curr_x, skeleton.shape)
        skeleton_neighbors = [(ny, nx) for ny, nx in neighbors if skeleton[ny, nx]]
        
        # Exclude the pixel we came from
        next_candidates = [p for p in skeleton_neighbors if p != prev_pixel]
        
        if len(next_candidates) == 0:
            # Dead end (no more skeleton)
            return None, None
        elif len(next_candidates) == 1:
            # Continue along path
            next_pixel = next_candidates[0]
            path.append(next_pixel)
            prev_pixel = current_pixel
            current_pixel = next_pixel
        else:
            # Multiple choices = we've reached a branch/node
            # This shouldn't happen in a proper skeleton, but handle it
            return path, None
    
    return None, None


def canonicalize_edge(start_node, end_node):
    """Create a canonical (symmetric) edge representation.
    
    Args:
        start_node, end_node: Pixel tuples
    
    Returns:
        Sorted tuple for undirected edge.
    """
    return tuple(sorted([start_node, end_node]))


def build_graph_from_skeleton(skeleton, junctions, endpoints, junction_centroids, labeled_junctions):
    """Build the complete graph structure.
    
    Args:
        skeleton: Boolean array
        junctions, endpoints: Sets of pixel tuples
        junction_centroids: Dict mapping label_id to centroid (y, x)
        labeled_junctions: Labeled connected component array
    
    Returns:
        (nodes_list, edges_list, pixel_to_node_dict)
    """
    # Trace edges
    edges_raw = trace_edges_from_nodes(skeleton, endpoints, junctions, 
                                       junction_centroids, labeled_junctions)
    
    # Build final node list
    nodes, pixel_to_node = build_node_list(endpoints, junctions, junction_centroids, labeled_junctions)
    
    # Deduplicate and finalize edges
    seen_edges = set()
    edges = []
    
    for edge_raw in edges_raw:
        start_pix = edge_raw['start']
        end_pix = edge_raw['end']
        canonical = canonicalize_edge(start_pix, end_pix)
        
        if canonical not in seen_edges:
            seen_edges.add(canonical)
            edges.append(edge_raw)
    
    return nodes, edges, pixel_to_node


def compute_edge_attributes(edges, nodes, pixel_to_node, labeled_junctions, junction_centroids):
    """Compute path_length, euclidean distance, and tortuosity for each edge.
    
    Args:
        edges: List of edge dicts from trace_edges_from_nodes
        nodes: List of (id, y, x, type) tuples
        pixel_to_node: Dict mapping (y, x) -> node_id
        labeled_junctions: Labeled array for junction pixels
        junction_centroids: Dict mapping label_id to centroid
    
    Returns:
        List of edge dicts with attributes added (self-loops removed).
    """
    # Create mapping from node pixel to node_id
    node_id_map = {}  # (y, x) -> node_id
    for node_id, y, x, node_type in nodes:
        # For junctions, map from the centroid location
        if node_type == 'junction':
            # Find which junction pixels map to this node
            pass
        node_id_map[(node_id, y, x)] = node_id
    
    # Simpler: use node coordinates directly
    node_coords = {}  # node_id -> (y, x)
    for node_id, y, x, node_type in nodes:
        node_coords[node_id] = (y, x)
    
    # Map each edge's start/end pixels to node_ids
    edges_with_attrs = []
    
    for edge in edges:
        start_pix = edge['start']
        end_pix = edge['end']
        path = edge['path']
        
        # Find node ids
        start_node_id = None
        end_node_id = None
        for node_id, y, x, node_type in nodes:
            if node_type == 'endpoint':
                if (int(y), int(x)) == start_pix:
                    start_node_id = node_id
                if (int(y), int(x)) == end_pix:
                    end_node_id = node_id
            elif node_type == 'junction':
                # Match to centroid (now snapped to skeleton pixel)
                if abs(y - start_pix[0]) < 0.5 and abs(x - start_pix[1]) < 0.5:
                    start_node_id = node_id
                if abs(y - end_pix[0]) < 0.5 and abs(x - end_pix[1]) < 0.5:
                    end_node_id = node_id
        
        if start_node_id is None or end_node_id is None:
            continue  # Skip edges without valid nodes
        
        # Skip self-loops
        if start_node_id == end_node_id:
            continue
        
        # Compute path length accounting for diagonal moves: sqrt(2) for diagonal, 1 for orthogonal
        # Path includes pixels from first neighbor to end node (not including start node itself)
        # So we need to add:
        # 1) Distance from start_node to first path pixel
        # 2) Distance along path
        # 3) Last path pixel should be at or near end_node
        
        path_length = 0.0
        
        # Distance from start node to first path pixel
        if len(path) > 0:
            y1, x1 = node_coords[start_node_id]
            y2, x2 = path[0]
            dy = abs(y2 - y1)
            dx = abs(x2 - x1)
            if (dy == 1 and dx == 0) or (dy == 0 and dx == 1):
                path_length += 1.0
            elif dy == 1 and dx == 1:
                path_length += np.sqrt(2)
            else:
                path_length += np.hypot(dy, dx)
        
        # Distance along the path
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            dy = abs(y2 - y1)
            dx = abs(x2 - x1)
            # Orthogonal move (dy=1, dx=0) or (dy=0, dx=1): distance = 1
            # Diagonal move (dy=1, dx=1): distance = sqrt(2)
            if (dy == 1 and dx == 0) or (dy == 0 and dx == 1):
                path_length += 1.0
            elif dy == 1 and dx == 1:
                path_length += np.sqrt(2)
            else:
                path_length += np.hypot(dy, dx)
        
        # Euclidean distance
        start_y, start_x = node_coords[start_node_id]
        end_y, end_x = node_coords[end_node_id]
        euclidean = np.sqrt((end_y - start_y)**2 + (end_x - start_x)**2)
        
        # Tortuosity
        tortuosity = path_length / max(euclidean, 1e-6)
        
        edges_with_attrs.append({
            'u': start_node_id,
            'v': end_node_id,
            'path_length': path_length,
            'euclidean': euclidean,
            'tortuosity': tortuosity,
            'path': path
        })
    
    return edges_with_attrs


def build_adjacency_matrices(num_nodes, edges, weight_attr='tortuosity'):
    """Build unweighted and weighted adjacency matrices.
    
    Args:
        num_nodes: Number of nodes
        edges: List of edge dicts with {u, v, tortuosity, path_length, euclidean}
        weight_attr: Which attribute to use as weight ('tortuosity', 'path_length', etc.)
    
    Returns:
        (A, W) where A is unweighted (0/1) and W is weighted adjacency matrix.
    """
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    W = np.zeros((num_nodes, num_nodes), dtype=float)
    
    for edge in edges:
        u, v = edge['u'], edge['v']
        
        # Skip self-loops
        if u == v:
            continue
        
        weight = edge.get(weight_attr, 1.0)
        
        # Undirected graph: set both (u, v) and (v, u)
        A[u, v] = 1
        A[v, u] = 1
        W[u, v] = weight
        W[v, u] = weight
    
    return A, W


def save_results(outdir, nodes, edges, A, W, weight_attr='tortuosity'):
    """Save results to CSV and NPY files.
    
    Args:
        outdir: Output directory
        nodes: List of (id, y, x, type) tuples
        edges: List of edge dicts
        A: Unweighted adjacency matrix
        W: Weighted adjacency matrix
        weight_attr: Name of weight attribute used
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Save nodes
    nodes_csv = os.path.join(outdir, 'nodes.csv')
    with open(nodes_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'y', 'x', 'type'])
        for node_id, y, x, node_type in nodes:
            writer.writerow([node_id, y, x, node_type])
    print(f"Saved: {nodes_csv}")
    
    # Save edges
    edges_csv = os.path.join(outdir, 'edges.csv')
    with open(edges_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u', 'v', 'path_length', 'euclidean', 'tortuosity', 'weight'])
        for edge in edges:
            writer.writerow([
                edge['u'],
                edge['v'],
                edge['path_length'],
                f"{edge['euclidean']:.4f}",
                f"{edge['tortuosity']:.4f}",
                f"{edge.get(weight_attr, 1.0):.4f}"
            ])
    print(f"Saved: {edges_csv}")
    
    # Save adjacency matrices
    A_npy = os.path.join(outdir, 'adjacency_unweighted.npy')
    W_npy = os.path.join(outdir, 'adjacency_weighted.npy')
    np.save(A_npy, A)
    np.save(W_npy, W)
    print(f"Saved: {A_npy}")
    print(f"Saved: {W_npy}")
    
    # Save metadata
    metadata_txt = os.path.join(outdir, 'metadata.txt')
    with open(metadata_txt, 'w') as f:
        f.write(f"Nodes: {len(nodes)}\n")
        f.write(f"Edges: {len(edges)}\n")
        f.write(f"Weight attribute: {weight_attr}\n")
    print(f"Saved: {metadata_txt}")


def print_graph_stats(skeleton, nodes, edges):
    """Print quality control statistics.
    
    Args:
        skeleton: Boolean array
        nodes: List of nodes
        edges: List of edges
    """
    # Connected components
    labeled, num_cc = ndimage.label(skeleton)
    
    num_endpoints = sum(1 for _, _, _, t in nodes if t == 'endpoint')
    num_junctions = sum(1 for _, _, _, t in nodes if t == 'junction')
    
    print("\n=== Graph Statistics ===")
    print(f"Total nodes: {len(nodes)}")
    print(f"  - Endpoints: {num_endpoints}")
    print(f"  - Junctions: {num_junctions}")
    print(f"Total edges: {len(edges)}")
    print(f"Skeleton connected components: {num_cc}")


def plot_graph_overlay(skeleton, nodes, edges, output_path=None):
    """Plot nodes and edges overlaid on skeleton (optional debug visualization).
    
    Args:
        skeleton: Boolean array
        nodes: List of (id, y, x, type) tuples
        edges: List of edge dicts
        output_path: Path to save figure (if provided)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Show skeleton
    ax.imshow(skeleton, cmap='gray', alpha=0.5)
    
    # Plot edges
    for edge in edges:
        path = edge['path']
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, 'b-', linewidth=0.5, alpha=0.3)
    
    # Plot nodes
    endpoints = [(y, x) for _, y, x, t in nodes if t == 'endpoint']
    junctions = [(y, x) for _, y, x, t in nodes if t == 'junction']
    
    if endpoints:
        ey, ex = zip(*endpoints)
        ax.scatter(ex, ey, c='green', s=20, label='Endpoints', zorder=5)
    
    if junctions:
        jy, jx = zip(*junctions)
        ax.scatter(jx, jy, c='red', s=20, label='Junctions', zorder=5)
    
    ax.legend()
    ax.set_title('Skeleton Graph: Nodes and Edges')
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert skeleton image to graph with adjacency matrices"
    )
    parser.add_argument('--image', required=True, help='Path to skeleton PNG')
    parser.add_argument('--outdir', required=True, help='Output directory')
    parser.add_argument('--weight', choices=['tortuosity', 'path_length', 'euclidean'], 
                       default='tortuosity', help='Edge weight attribute')
    parser.add_argument('--plot', action='store_true', help='Generate debug plot')
    args = parser.parse_args()
    
    print(f"Loading skeleton: {args.image}")
    skeleton = load_skeleton(args.image)
    
    print("Detecting nodes...")
    node_pixels = detect_node_pixels(skeleton)
    endpoints = node_pixels['endpoints']
    junctions = node_pixels['junctions']
    
    print(f"  Endpoints: {len(endpoints)}, Junctions: {len(junctions)}")
    
    print("Compressing junction clusters...")
    junction_centroids, junction_mask, labeled_junctions = compress_junction_clusters(junctions, skeleton)
    print(f"  Compressed to {len(junction_centroids)} junction clusters")
    
    print("Building graph...")
    nodes, edges, pixel_to_node = build_graph_from_skeleton(
        skeleton, junctions, endpoints, junction_centroids, labeled_junctions
    )
    
    print("Computing edge attributes...")
    edges = compute_edge_attributes(edges, nodes, pixel_to_node, labeled_junctions, junction_centroids)
    
    print("Building adjacency matrices...")
    num_nodes = len(nodes)
    A, W = build_adjacency_matrices(num_nodes, edges, weight_attr=args.weight)
    
    print("Saving results...")
    save_results(args.outdir, nodes, edges, A, W, weight_attr=args.weight)
    
    print_graph_stats(skeleton, nodes, edges)
    
    if args.plot:
        plot_overlay_path = os.path.join(args.outdir, 'graph_overlay.png')
        plot_graph_overlay(skeleton, nodes, edges, output_path=plot_overlay_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
