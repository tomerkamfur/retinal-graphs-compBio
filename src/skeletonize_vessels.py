'''
Create a 1-pixel-wide medial-axis skeleton from a filled vessel image.

Pipeline:
1) Load image with skimage.io.imread and convert to grayscale if needed.
2) Create a FILLED vessel mask using Otsu or adaptive thresholding.
3) Morphological cleanup: remove small objects, fill small holes, optional closing.
4) Skeletonization: use skimage.morphology.skeletonize on boolean mask.
5) Post-checks: ensure skeleton pixels are inside vessel mask and detect 2x2 thick blocks.
6) Save `vessel_mask.png`, `skeleton.png`, and `overlay.png`.

Usage:
    python src/skeletonize_vessels.py --input path/to/image.png --outdir outputs

Requirements: numpy, scikit-image
'''
import argparse
import os
import numpy as np
from skimage import io, color, img_as_float, img_as_ubyte
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk, square, skeletonize, erosion


def load_image(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img_as_float(img)
    return img


def make_filled_mask(img, method="otsu", block_size=35, offset=0.0):
    '''
    Create a filled boolean vessel mask (True = vessel interior).
    method: 'otsu' or 'adaptive'
    block_size, offset: used only for adaptive thresholding
    '''
    if method == "otsu":
        thr = threshold_otsu(img)
        mask = img >= thr
    else:
        thr = threshold_local(img, block_size=block_size, offset=offset)
        mask = img >= thr
    return mask.astype(bool)


def morphological_cleanup(mask, min_size=150, hole_size=150, closing_radius=5):
    '''
    Remove small objects, fill small holes, and optionally close small gaps.
    After this step the vessels should be solid, filled regions.
    '''
    cleaned = remove_small_objects(mask, min_size=min_size)
    cleaned = remove_small_holes(cleaned, area_threshold=hole_size)
    if closing_radius and closing_radius > 0:
        selem = disk(closing_radius)
        cleaned = binary_closing(cleaned, selem)
    # Ensure boolean
    return cleaned.astype(bool)


def enforce_single_pixel_width(skel):
    '''
    Remove pixels from 2x2 blocks to enforce strict 1-pixel-wide skeleton.
    Detects 2x2 all-True blocks and removes pixels to break the block
    while preserving connectivity.
    '''
    result = skel.copy()
    height, width = skel.shape
    
    # Scan for 2x2 blocks and break them
    changed = True
    iterations = 0
    max_iterations = 10
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        # Detect 2x2 blocks
        if height > 1 and width > 1:
            a = result[:-1, :-1] & result[1:, :-1] & result[:-1, 1:] & result[1:, 1:]
            if np.any(a):
                changed = True
                # For each 2x2 block, remove the lower-right pixel
                to_remove = np.zeros_like(result, dtype=bool)
                for i in range(height - 1):
                    for j in range(width - 1):
                        if a[i, j]:
                            to_remove[i+1, j+1] = True
                result = result & (~to_remove)
    
    return result.astype(bool)


def compute_skeleton(vessel_mask):
    '''
    Skeletonize a filled boolean vessel mask using skimage.morphology.skeletonize.
    Input must be boolean; output is boolean skeleton with single-pixel-wide centerlines.
    Then enforce strict 1-pixel width by removing 2x2 blocks.
    '''
    skel = skeletonize(vessel_mask)
    skel = enforce_single_pixel_width(skel)
    return skel.astype(bool)


def check_skeleton(skel, vessel_mask):
    '''
    Run sanity checks on skeleton.
    - Ensure skeleton pixels lie inside vessel_mask.
    - Detect any 2x2 blocks of True in the skeleton (indicates >1-pixel-thick areas).
    Returns a dict of checks and boolean pass/fail.
    '''
    issues = {}
    # Check skeleton is subset of vessel_mask
    outside = np.any(skel & (~vessel_mask))
    issues['skeleton_outside_vessel'] = bool(outside)

    # Detect any 2x2 all-True blocks (a simple test for >1-pixel-wide chunks)
    if skel.size == 0:
        issues['two_by_two_blocks'] = False
    else:
        a = skel[:-1, :-1] & skel[1:, :-1] & skel[:-1, 1:] & skel[1:, 1:]
        issues['two_by_two_blocks'] = bool(np.any(a))

    # Quick thickness check via 3x3 erosion: if any pixel remains after erosion,
    # that indicates there exist 3x3 full-True neighborhoods (rare) — keep as extra info.
    eroded = erosion(skel, square(3))
    issues['has_3x3_all_true'] = bool(np.any(eroded))

    # Count skeleton pixels
    issues['skeleton_pixel_count'] = int(np.count_nonzero(skel))
    return issues


def make_overlay(original_img, skel):
    '''
    Create an RGB overlay showing skeleton in red over the original grayscale image.
    original_img expected float [0,1] grayscale.
    '''
    if original_img.ndim == 2:
        gray = np.clip(original_img, 0.0, 1.0)
        rgb = np.dstack([gray, gray, gray])
    else:
        rgb = original_img.copy()
        if rgb.dtype != float:
            rgb = img_as_float(rgb)

    # Paint skeleton pixels red (R=1,G=0,B=0)
    overlay = rgb.copy()
    overlay[skel, 0] = 1.0
    overlay[skel, 1] = 0.0
    overlay[skel, 2] = 0.0
    return np.clip(overlay, 0.0, 1.0)


def save_binary_image(path, arr_bool):
    # Save boolean as 0/255 uint8 image
    img = (arr_bool.astype(np.uint8) * 255)
    io.imsave(path, img)


def main():
    parser = argparse.ArgumentParser(description="Create 1-pixel-wide skeleton from filled vessel image")
    parser.add_argument("--input", required=True, help="Path to input image (black background, white vessels)")
    parser.add_argument("--outdir", default=".", help="Directory to write outputs")
    parser.add_argument("--threshold", choices=['otsu', 'adaptive'], default='otsu')
    parser.add_argument("--min_size", type=int, default=150, help="Min size to keep objects (pixels)")
    parser.add_argument("--hole_size", type=int, default=150, help="Max hole area to fill (pixels)")
    parser.add_argument("--closing_radius", type=int, default=5, help="Disk radius for closing (0 to skip)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img = load_image(args.input)

    vessel_mask = make_filled_mask(img, method=args.threshold)
    vessel_mask = morphological_cleanup(vessel_mask, min_size=args.min_size, hole_size=args.hole_size,
                                        closing_radius=args.closing_radius)

    skeleton = compute_skeleton(vessel_mask)

    checks = check_skeleton(skeleton, vessel_mask)

    # Save outputs
    vessel_path = os.path.join(args.outdir, 'vessel_mask.png')
    skeleton_path = os.path.join(args.outdir, 'skeleton.png')
    overlay_path = os.path.join(args.outdir, 'overlay.png')

    save_binary_image(vessel_path, vessel_mask)
    save_binary_image(skeleton_path, skeleton)

    overlay = make_overlay(img, skeleton)
    io.imsave(overlay_path, img_as_ubyte(overlay))

    # Print concise report
    print('Saved:', vessel_path)
    print('Saved:', skeleton_path)
    print('Saved:', overlay_path)
    print('\nSkeleton sanity checks:')
    print(f"- Skeleton pixels: {checks['skeleton_pixel_count']}")
    if checks['skeleton_outside_vessel']:
        print("- WARNING: Skeleton has pixels outside the vessel mask (this should not happen).")
    else:
        print("- OK: Skeleton pixels lie inside the vessel mask.")

    if checks['two_by_two_blocks']:
        print("- WARNING: Detected 2x2 blocks in skeleton (possible >1-pixel-wide segments).")
    else:
        print("- OK: No 2x2 thick blocks detected in skeleton.")

    if checks['has_3x3_all_true']:
        print("- NOTE: Some 3x3 all-true neighborhoods exist in the skeleton (rare).")

    print('\nIf the skeleton looks like outlines or parallel lines, ensure the input to skeletonize is a FILLED binary mask.')


if __name__ == '__main__':
    main()
