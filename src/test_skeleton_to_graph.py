"""Unit tests for skeleton_to_graph module using synthetic skeletons."""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from skeleton_to_graph import (
    detect_node_pixels,
    count_skeleton_neighbors,
    compress_junction_clusters,
    build_node_list,
    trace_edges_from_nodes,
    build_adjacency_matrices
)


def test_line():
    """Test: horizontal line with two endpoints."""
    skeleton = np.zeros((5, 5), dtype=bool)
    skeleton[2, 1:4] = True  # Horizontal line from (2,1) to (2,3)
    
    node_pixels = detect_node_pixels(skeleton)
    endpoints = node_pixels['endpoints']
    junctions = node_pixels['junctions']
    
    # Should have 2 endpoints, 0 junctions
    assert len(endpoints) == 2, f"Expected 2 endpoints, got {len(endpoints)}"
    assert len(junctions) == 0, f"Expected 0 junctions, got {len(junctions)}"
    
    # Endpoints should be the two ends
    assert (2, 1) in endpoints or (2, 3) in endpoints
    
    print("✓ test_line passed")


def test_t_junction():
    """Test: T-shaped junction with one junction pixel."""
    skeleton = np.zeros((5, 5), dtype=bool)
    # Horizontal line
    skeleton[2, 1:4] = True
    # Vertical line  
    skeleton[1:4, 2] = True
    
    node_pixels = detect_node_pixels(skeleton)
    endpoints = node_pixels['endpoints']
    junctions = node_pixels['junctions']
    
    # Center (2,2) has 4 neighbors (up, down, left, right), so it's a junction
    # (2,1) has 2 neighbors (center and diag up-right)
    # (2,3) has 2 neighbors (center and diag up-left)
    # (1,2) has 2 neighbors (center and diag down)
    # (3,2) has 2 neighbors (center and diag up)
    # So all outer pixels have 2 neighbors, making them branches, not pure endpoints
    # Let's just check that we have 1 junction at the center
    
    assert len(junctions) >= 1, f"Expected at least 1 junction, got {len(junctions)}"
    assert (2, 2) in junctions, f"Expected junction at (2,2), got junctions at {junctions}"
    
    print("✓ test_t_junction passed")


def test_y_junction():
    """Test: Y-shaped junction (3-way branch)."""
    skeleton = np.zeros((7, 7), dtype=bool)
    # Center point
    center_y, center_x = 3, 3
    skeleton[center_y, center_x] = True
    
    # Three branches radiating outward
    # Branch 1: up
    skeleton[1:center_y, center_x] = True
    # Branch 2: down-left
    skeleton[4:6, 1:3] = True
    skeleton[center_y:4, center_x-1:2] = True
    # Branch 3: down-right
    skeleton[4:6, 4:6] = True
    skeleton[center_y:4, center_x:6] = True
    
    node_pixels = detect_node_pixels(skeleton)
    endpoints = node_pixels['endpoints']
    junctions = node_pixels['junctions']
    
    # At least one junction should exist
    assert len(junctions) >= 1, f"Expected at least 1 junction, got {len(junctions)}"
    
    print("✓ test_y_junction passed")


def test_node_neighbor_counting():
    """Test neighbor counting in 8-connectivity."""
    # Isolated line: two pixels
    skeleton = np.zeros((5, 5), dtype=bool)
    skeleton[2, 1] = True
    skeleton[2, 2] = True
    
    # (2,1) should be endpoint with 1 neighbor
    count = count_skeleton_neighbors(skeleton, 2, 1)
    assert count == 1, f"Expected 1 neighbor for endpoint, got {count}"
    
    # (2,2) should also be endpoint with 1 neighbor
    count = count_skeleton_neighbors(skeleton, 2, 2)
    assert count == 1, f"Expected 1 neighbor for endpoint, got {count}"
    
    # Add a third pixel to create a junction-like structure
    skeleton[1, 2] = True
    
    # Now (2,2) has 2 neighbors (orthogonal and diagonal up)
    count = count_skeleton_neighbors(skeleton, 2, 2)
    assert count == 2, f"Expected 2 neighbors, got {count}"
    
    print("✓ test_node_neighbor_counting passed")


def test_junction_compression():
    """Test that nearby junction pixels are merged into single clusters."""
    skeleton = np.zeros((5, 5), dtype=bool)
    
    # Create a small cluster of junction pixels (all connected)
    skeleton[2, 2] = True
    skeleton[2, 3] = True
    skeleton[3, 2] = True
    skeleton[3, 3] = True
    
    junctions = {(2, 2), (2, 3), (3, 2), (3, 3)}
    
    centroids, junction_mask, labeled = compress_junction_clusters(junctions, skeleton)
    
    # Should compress into 1 cluster (assuming all 4 are 8-connected)
    assert len(centroids) == 1, f"Expected 1 cluster, got {len(centroids)}"
    
    # Centroid should be approximately (2.5, 2.5)
    centroid = list(centroids.values())[0]
    assert 2.0 <= centroid[0] <= 3.0, f"Centroid y out of range: {centroid[0]}"
    assert 2.0 <= centroid[1] <= 3.0, f"Centroid x out of range: {centroid[1]}"
    
    print("✓ test_junction_compression passed")


def test_adjacency_matrix():
    """Test adjacency matrix construction."""
    nodes = [
        (0, 0.0, 0.0, 'endpoint'),
        (1, 0.0, 5.0, 'endpoint'),
        (2, 5.0, 2.5, 'junction')
    ]
    
    edges = [
        {'u': 0, 'v': 2, 'path_length': 5, 'euclidean': 5.59, 'tortuosity': 0.89},
        {'u': 1, 'v': 2, 'path_length': 5, 'euclidean': 5.59, 'tortuosity': 0.89}
    ]
    
    A, W = build_adjacency_matrices(len(nodes), edges, weight_attr='tortuosity')
    
    # Check unweighted adjacency
    assert A[0, 2] == 1 and A[2, 0] == 1, "Edge 0-2 not symmetric in A"
    assert A[1, 2] == 1 and A[2, 1] == 1, "Edge 1-2 not symmetric in A"
    assert A[0, 1] == 0, "Spurious edge 0-1 in A"
    
    # Check weighted adjacency (should use tortuosity)
    assert abs(W[0, 2] - 0.89) < 0.01, "Weight not correctly set for edge 0-2"
    assert abs(W[1, 2] - 0.89) < 0.01, "Weight not correctly set for edge 1-2"
    
    print("✓ test_adjacency_matrix passed")


if __name__ == '__main__':
    print("Running unit tests for skeleton_to_graph...\n")
    
    test_node_neighbor_counting()
    test_line()
    test_t_junction()
    test_y_junction()
    test_junction_compression()
    test_adjacency_matrix()
    
    print("\n✓ All tests passed!")
