"""Main pipeline: Process all final_mask images through complete graph extraction and analysis.

For each folder in data/picturese_for_graphs:
  1. Find *_final_mask image
  2. Skeletonize the mask
  3. Convert skeleton to graph (nodes, edges, adjacency matrices)
  4. Compute pairwise endpoint path statistics
  5. Create graph overlay with statistics
  6. Save all outputs to the folder

Usage:
    python main_pipeline.py [--data-dir data/picturese_for_graphs] [--verbose]

Outputs per folder:
  - skeleton.png: 1-pixel-wide vessel centerline
  - nodes.csv, edges.csv: Node and edge lists
  - adjacency_unweighted.npy, adjacency_weighted.npy: Adjacency matrices
  - metadata.txt: Graph statistics
  - graph_overlay.png: Visualization of graph on skeleton
  - endpoint_pair_paths.csv: Pairwise shortest path statistics
  - graph_overlay_stats.png: Graph visualization with statistics
"""

import argparse
import os
import sys
import glob
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from skeletonize_vessels import main as skeletonize_main
from skeleton_to_graph import main as graph_main
import subprocess
import shutil


def find_final_mask(folder):
    """Find *_final_mask image in folder.
    
    Returns:
        Path to final_mask image, or None if not found.
    """
    patterns = ['*_final_mask.png', '*_final_mask.jpg', '*_final_mask.tif']
    for pattern in patterns:
        matches = glob.glob(os.path.join(folder, pattern))
        if matches:
            return matches[0]
    return None


def run_skeletonization(mask_image, output_dir, verbose=False):
    """Run skeletonization on the mask image.
    
    Args:
        mask_image: Path to final_mask image
        output_dir: Directory to save skeleton outputs
        verbose: Print debug info
    
    Returns:
        Path to skeleton.png, or None on error
    """
    if verbose:
        print(f"  Skeletonizing: {mask_image}")
    
    try:
        # Create temporary args for skeletonize_main
        sys.argv = [
            'skeletonize_vessels.py',
            '--input', mask_image,
            '--outdir', output_dir,
            '--threshold', 'otsu',
            '--min_size', '5',
            '--hole_size', '5',
            '--closing_radius', '2'
        ]
        skeletonize_main()
        
        skeleton_path = os.path.join(output_dir, 'skeleton.png')
        if os.path.exists(skeleton_path):
            if verbose:
                print(f"    ✓ Skeleton saved: {skeleton_path}")
            return skeleton_path
        else:
            print(f"    ✗ Skeleton not found at {skeleton_path}")
            return None
    except Exception as e:
        print(f"    ✗ Skeletonization failed: {e}")
        return None


def run_graph_extraction(skeleton_path, output_dir, verbose=False):
    """Convert skeleton to graph.
    
    Args:
        skeleton_path: Path to skeleton.png
        output_dir: Directory to save graph outputs
        verbose: Print debug info
    
    Returns:
        True on success, False on error
    """
    if verbose:
        print(f"  Extracting graph from skeleton")
    
    try:
        sys.argv = [
            'skeleton_to_graph.py',
            '--image', skeleton_path,
            '--outdir', output_dir,
            '--weight', 'path_length',
            '--plot'
        ]
        graph_main()
        
        # Check that key files exist
        required_files = ['nodes.csv', 'edges.csv', 'adjacency_unweighted.npy', 'adjacency_weighted.npy']
        all_exist = all(os.path.exists(os.path.join(output_dir, f)) for f in required_files)
        
        if all_exist:
            if verbose:
                print(f"    ✓ Graph extracted successfully")
            return True
        else:
            print(f"    ✗ Some graph files missing")
            return False
    except Exception as e:
        print(f"    ✗ Graph extraction failed: {e}")
        return None


def run_path_statistics(graph_dir, output_dir, verbose=False):
    """Compute pairwise endpoint path statistics.
    
    Args:
        graph_dir: Directory with adjacency matrices and CSVs
        output_dir: Directory to save path statistics
        verbose: Print debug info
    
    Returns:
        True on success, False on error
    """
    if verbose:
        print(f"  Computing path statistics")
    
    try:
        # Import and run directly
        from graph_path_stats import load_graph_data, build_nx_graph, pairwise_endpoint_paths, summarize_pairwise_results, tortuosity_edge_statistics
        
        nodes, endpoints, node_coords, A, W, edge_attr_map = load_graph_data(graph_dir)
        G = build_nx_graph(A, W, edge_attr_map=edge_attr_map)
        
        results = pairwise_endpoint_paths(G, endpoints, node_coords, edge_attr_map, weight_attr='weight')
        summary = summarize_pairwise_results(results, outdir=output_dir)
        edge_t_stats = tortuosity_edge_statistics(edge_attr_map)
        
        if verbose:
            print(f"    ✓ Computed {len(results)} pairwise paths")
            print(f"    ✓ Path statistics saved")
        
        return True
    except Exception as e:
        print(f"    ✗ Path statistics failed: {e}")
        return False


def run_overlay_with_stats(skeleton_path, graph_dir, stats_dir, output_dir, verbose=False):
    """Create graph overlay with statistics visualization.
    
    Args:
        skeleton_path: Path to skeleton.png
        graph_dir: Directory with graph files
        stats_dir: Directory with path statistics
        output_dir: Directory to save overlay
        verbose: Print debug info
    
    Returns:
        True on success, False on error
    """
    if verbose:
        print(f"  Creating overlay with statistics")
    
    try:
        from overlay_with_stats import main as overlay_main
        
        overlay_img = os.path.join(graph_dir, 'graph_overlay.png')
        pairwise_npy = os.path.join(stats_dir, 'endpoint_pair_paths.npy')
        edges_csv = os.path.join(graph_dir, 'edges.csv')
        metadata_txt = os.path.join(graph_dir, 'metadata.txt')
        output_img = os.path.join(output_dir, 'graph_overlay_stats.png')
        
        sys.argv = [
            'overlay_with_stats.py',
            '--overlay', overlay_img,
            '--pairwise', pairwise_npy,
            '--edges', edges_csv,
            '--metadata', metadata_txt,
            '--out', output_img
        ]
        overlay_main()
        
        if os.path.exists(output_img):
            if verbose:
                print(f"    ✓ Overlay saved: {output_img}")
            return True
        else:
            print(f"    ✗ Overlay file not created")
            return False
    except Exception as e:
        print(f"    ✗ Overlay creation failed: {e}")
        return False


def process_folder(folder_path, verbose=False):
    """Process a single folder through the complete pipeline.
    
    Args:
        folder_path: Path to folder containing final_mask image
        verbose: Print debug info
    
    Returns:
        Dictionary with processing status
    """
    folder_name = os.path.basename(folder_path)
    print(f"\n{'='*70}")
    print(f"Processing: {folder_name}")
    print(f"{'='*70}")
    
    result = {
        'folder': folder_name,
        'path': folder_path,
        'status': 'failed',
        'steps': {}
    }
    
    # Step 1: Find final_mask
    mask_image = find_final_mask(folder_path)
    if not mask_image:
        print(f"✗ No *_final_mask image found in {folder_path}")
        return result
    
    print(f"Found final_mask: {os.path.basename(mask_image)}")
    result['mask_file'] = mask_image
    
    # Use the folder itself as output directory (save everything there)
    output_dir = folder_path
    
    # Step 2: Skeletonization
    print(f"\n[1/4] Skeletonization...")
    skeleton_path = run_skeletonization(mask_image, output_dir, verbose)
    result['steps']['skeletonization'] = 'success' if skeleton_path else 'failed'
    if not skeleton_path:
        return result
    
    # Step 3: Graph extraction
    print(f"\n[2/4] Graph Extraction...")
    success = run_graph_extraction(skeleton_path, output_dir, verbose)
    result['steps']['graph_extraction'] = 'success' if success else 'failed'
    if not success:
        return result
    
    # Step 4: Path statistics
    print(f"\n[3/4] Path Statistics...")
    success = run_path_statistics(output_dir, output_dir, verbose)
    result['steps']['path_statistics'] = 'success' if success else 'failed'
    if not success:
        return result
    
    # Step 5: Overlay with stats
    print(f"\n[4/4] Creating Overlay...")
    success = run_overlay_with_stats(skeleton_path, output_dir, output_dir, output_dir, verbose)
    result['steps']['overlay_stats'] = 'success' if success else 'failed'
    if not success:
        return result
    
    result['status'] = 'success'
    print(f"\n✓ {folder_name} completed successfully!")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run complete pipeline on all final_mask images'
    )
    parser.add_argument(
        '--data-dir',
        default='data/picturese_for_graphs',
        help='Directory containing image folders'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose debug information'
    )
    parser.add_argument(
        '--folders',
        nargs='+',
        help='Process specific folders (by name) instead of all'
    )
    
    args = parser.parse_args()
    
    # Find all folders
    if not os.path.exists(args.data_dir):
        print(f"✗ Data directory not found: {args.data_dir}")
        return 1
    
    all_folders = [
        d for d in glob.glob(os.path.join(args.data_dir, '*'))
        if os.path.isdir(d)
    ]
    
    if not all_folders:
        print(f"✗ No folders found in {args.data_dir}")
        return 1
    
    # Filter by name if specified
    if args.folders:
        all_folders = [
            f for f in all_folders
            if any(name in os.path.basename(f) for name in args.folders)
        ]
    
    print(f"Found {len(all_folders)} folders to process")
    print(f"Data directory: {args.data_dir}")
    
    # Process each folder
    results = []
    for folder_path in sorted(all_folders):
        result = process_folder(folder_path, verbose=args.verbose)
        results.append(result)
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\nTotal folders: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed folders:")
        for r in results:
            if r['status'] != 'success':
                print(f"  - {r['folder']}: {r['steps']}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
