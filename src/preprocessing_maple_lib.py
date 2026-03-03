'''
Vessel segmentation using fundus-image-toolbox (MAPLES-like pipeline).

Outputs binary masks:
- vessels: 255
- background: 0

Examples:
    python src/preprocessing_maple_lib.py --input data/messidor-2/20051020_55701_0100_PP.png --output results/masks_maple/20051020_55701_0100_PP_vessels.png
    python src/preprocessing_maple_lib.py --input-dir data/messidor-2 --output-dir results/masks_maple
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retinal vessel segmentation with fundus-image-toolbox")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="Single image path")
    mode.add_argument("--input-dir", type=Path, help="Directory with images")

    parser.add_argument("--output", type=Path, help="Output mask path in single-image mode")
    parser.add_argument("--output-dir", type=Path, help="Output folder in batch mode")
    parser.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"])

    parser.add_argument("--resize", type=int, default=1500, help="Square resize before inference")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto, cpu, or cuda:0")
    parser.add_argument("--model-threshold", type=float, default=0.5, help="Base threshold used inside toolbox model voting")
    parser.add_argument("--threshold", type=float, default=0.25, help="Base probability threshold")
    parser.add_argument("--auto-threshold", action="store_true", help="Choose threshold per-image from probability distribution")
    parser.add_argument("--target-frac-min", type=float, default=0.10, help="Auto mode min vessel fraction")
    parser.add_argument("--target-frac-max", type=float, default=0.25, help="Auto mode max vessel fraction")
    parser.add_argument("--min-area", type=int, default=80, help="Remove connected components smaller than this area")
    parser.add_argument("--close-kernel", type=int, default=3, help="Morph close kernel size")
    parser.add_argument("--open-kernel", type=int, default=0, help="Morph open kernel size; 0 disables opening")
    return parser.parse_args()


def load_maple_tools():
    try:
        from fundus_image_toolbox import load_segmentation_ensemble, ensemble_predict_segmentation
        from fundus_image_toolbox.circle_crop import crop as circle_crop
    except Exception as e:
        raise ImportError(
            "fundus_image_toolbox is required. Install it in your environment before running this script."
        ) from e
    return load_segmentation_ensemble, ensemble_predict_segmentation, circle_crop


def ensure_prob_2d(prob: np.ndarray) -> np.ndarray:
    '''
    Ensure the probability map is 2D with shape (H, W).
    Handles cases where the input may have extra dimensions.
    '''
    p = np.asarray(prob, dtype=np.float32)
    if p.ndim == 3:
        if p.shape[-1] == 1:
            p = p[..., 0]
        elif p.shape[0] == 1:
            p = p[0, ...]
        else:
            p = p[..., 0]
    if p.ndim != 2:
        raise ValueError(f"Unexpected probability shape: {p.shape}")
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)
    return p


def choose_threshold_auto(prob: np.ndarray, frac_min: float, frac_max: float) -> float:
    '''
    select a threshold to achieve a vessel fraction between frac_min and frac_max.
    If the distribution allows, choose a threshold that gets close to the midpoint of that range.
    '''
    vals = prob[np.isfinite(prob)]
    if vals.size == 0:
        return 0.25

    target_mid = 0.5 * (frac_min + frac_max)
    candidates = np.linspace(0.12, 0.60, 25)
    best_t = 0.25
    best_score = float("inf")
    for t in candidates:
        frac = float(np.mean(prob >= t))
        if frac_min <= frac <= frac_max:
            score = abs(frac - target_mid)
        else:
            score = min(abs(frac - frac_min), abs(frac - frac_max)) + 0.2
        if score < best_score:
            best_score = score
            best_t = float(t)
    return best_t


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    '''
    remove connected components smaller than min_area from the binary mask.
    '''
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def postprocess(prob: np.ndarray, threshold: float, min_area: int,
    close_kernel: int, open_kernel: int,) -> np.ndarray:
    '''
    post-process the probability map to create a binary vessel mask:
    '''
    mask = (prob >= threshold).astype(np.uint8)

    if close_kernel > 0:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if open_kernel > 0:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    mask = remove_small_components(mask, min_area=min_area)
    return (mask * 255).astype(np.uint8)


def is_image(path: Path, exts: Iterable[str]) -> bool:
    return path.is_file() and path.suffix.lower() in {e.lower() for e in exts}


def run_single(image_path: Path, output_path: Path, ensemble_models, predict_fn,
    circle_crop, args: argparse.Namespace,) -> None:
    '''
    run vessel segmentation on a single image and save the output mask.
    '''
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = circle_crop(img, size=(args.resize, args.resize))
    img = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

    prob = ensure_prob_2d(
        predict_fn(
            ensemble_models=ensemble_models,
            images=img,
            device=args.device,
            threshold=args.model_threshold,
        )
    )
    thr = (
        choose_threshold_auto(prob, args.target_frac_min, args.target_frac_max)
        if args.auto_threshold
        else args.threshold
    )
    mask = postprocess(prob, thr, args.min_area, args.close_kernel, args.open_kernel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)

    frac = float(np.mean(mask > 0))
    print(f"Saved: {output_path}")
    print(f"Threshold={thr:.3f} | mask_fraction={frac:.4f}")


def run_batch(input_dir: Path, output_dir: Path, ensemble_models, predict_fn, circle_crop, args: argparse.Namespace) -> None:
    '''
    run vessel segmentation on all images in the input directory and save output masks to the output directory.
    '''
    images = [p for p in sorted(input_dir.rglob("*")) if is_image(p, args.exts)]
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for p in tqdm(images, desc="Segmenting"):
        out = output_dir / f"{p.stem}_vessels.png"
        run_single(p, out, ensemble_models, predict_fn, circle_crop, args)


def main() -> None:
    args = parse_args()
    load_ensemble, predict_fn, circle_crop = load_maple_tools()

    if args.device == "auto":
        try:
            import torch

            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    ensemble_models = load_ensemble(device=args.device)

    if args.input is not None:
        if args.output is None:
            raise ValueError("Single-image mode requires --output.")
        run_single(args.input, args.output, ensemble_models, predict_fn, circle_crop, args)
        return

    if args.output_dir is None:
        raise ValueError("Batch mode requires --output-dir.")
    run_batch(args.input_dir, args.output_dir, ensemble_models, predict_fn, circle_crop, args)


if __name__ == "__main__":
    main()
