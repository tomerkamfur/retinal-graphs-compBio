'''
Vessel segmentation from retinal fundus image to binary mask.

Output mask format:
- vessel pixels: 255 (white)
- background: 0 (black)

Usage examples:
    python src/preprocessing.py --input data/messidor-2/IM003360.JPG --output results/masks/IM003360_mask.png
    python src/preprocessing.py --input-dir data/messidor-2 --output-dir results/masks
'''

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from skimage import exposure, measure, morphology
from skimage.filters import frangi, gaussian
from skimage.morphology import disk


@dataclass
class SegmentationConfig:
    # FOV
    fov_threshold: int = 10
    inner_fov_fraction: float = 0.06
    inner_fov_min_band: int = 20

    # Bright artifacts
    disc_percentile: float = 99.2
    exudate_offset: float = 0.10
    inpaint_radius: int = 7

    # Vesselness + thresholding
    use_auto_thresholds: bool = True
    hi_percentile: float = 95.0
    lo_percentile: float = 82.0
    auto_hi_candidates: tuple[float, ...] = (99, 98, 97, 96, 95, 94, 93)
    auto_lo_candidates: tuple[float, ...] = (90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68)
    target_vessel_frac_min: float = 0.08
    target_vessel_frac_max: float = 0.25
    min_component_size: int = 80


def to_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if np.max(x) > 1.0:
        x /= 255.0
    return np.clip(x, 0.0, 1.0)


def imread_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    '''Compatibility wrapper for skimage API changes.

    Newer versions deprecate `min_size` in favor of `max_size` with inclusive
    behavior. Using `max_size=min_size-1` approximates old behavior.
    '''
    try:
        return morphology.remove_small_objects(mask, max_size=max(0, min_size - 1))
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=min_size)


def compute_fov_and_radius(rgb: np.ndarray, threshold: int) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    m = m.astype(bool)

    lab = measure.label(m)
    if lab.max() == 0:
        fov = np.ones(gray.shape, dtype=bool)
    else:
        largest = max(measure.regionprops(lab), key=lambda r: r.area)
        fov = lab == largest.label

    area = float(np.sum(fov))
    radius = float(np.sqrt(area / np.pi))
    return fov, radius


def inner_fov_mask(fov: np.ndarray, radius: float, frac: float, min_band: int) -> np.ndarray:
    band = int(max(min_band, frac * radius))
    return morphology.erosion(fov, disk(max(1, band)))


def preprocess_green(rgb: np.ndarray, fov_in: np.ndarray, radius: float) -> np.ndarray:
    g = to_float01(rgb[..., 1])
    g2 = g.copy()
    g2[~fov_in] = np.median(g[fov_in])

    p1, p99 = np.percentile(g2[fov_in], [1, 99])
    g2 = np.clip((g2 - p1) / (p99 - p1 + 1e-8), 0, 1)

    ks = int(max(32, radius * 0.20))
    g_eq = exposure.equalize_adapthist(g2, kernel_size=ks, clip_limit=0.02)

    sigma_bg = max(10.0, 0.04 * radius)
    bg = gaussian(g_eq, sigma=sigma_bg)
    I = g_eq - bg

    I = (I - np.min(I[fov_in])) / (np.ptp(I[fov_in]) + 1e-8)
    I[~fov_in] = 0.0
    return I


def detect_optic_disc(g: np.ndarray, fov_in: np.ndarray, radius: float, percentile: float) -> np.ndarray:
    g_s = gaussian(g, sigma=max(2.0, radius * 0.03))
    t = np.percentile(g_s[fov_in], percentile)
    disc = (g_s > t) & fov_in
    disc = morphology.closing(disc, disk(int(max(3, radius * 0.01))))
    disc = remove_small_components(disc, min_size=int(max(200, radius * 0.01)))

    lab = measure.label(disc)
    if lab.max() == 0:
        return np.zeros_like(disc, dtype=bool)
    largest = max(measure.regionprops(lab), key=lambda r: r.area)
    disc_mask = lab == largest.label
    disc_mask = morphology.dilation(disc_mask, disk(int(max(4, radius * 0.02))))
    return disc_mask


def detect_exudates(g: np.ndarray, fov_in: np.ndarray, radius: float, offset: float) -> np.ndarray:
    local_avg = gaussian(g, sigma=max(2.0, radius * 0.05))
    bright = (g > (local_avg + offset)) & fov_in
    bright = morphology.dilation(bright, disk(int(max(2, radius * 0.02))))
    bright = remove_small_components(bright, min_size=int(max(20, radius * 0.002)))
    return bright


def inpaint_mask(g: np.ndarray, mask: np.ndarray, inpaint_radius: int) -> np.ndarray:
    g_u8 = (g * 255).astype(np.uint8)
    m_u8 = (mask.astype(np.uint8) * 255)
    out = cv2.inpaint(g_u8, m_u8, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    return to_float01(out)


def disc_ring(disc_mask: np.ndarray, radius: float) -> np.ndarray:
    outer = morphology.dilation(disc_mask, disk(int(max(6, radius * 0.03))))
    inner = morphology.erosion(disc_mask, disk(int(max(2, radius * 0.01))))
    return outer & (~inner)


def vessel_probability(I: np.ndarray, fov_in: np.ndarray, radius: float, disc_mask: np.ndarray) -> np.ndarray:
    bg = gaussian(I, sigma=max(4.0, radius * 0.10))
    I_hp = np.clip(bg - I, 0, 1)
    I_hp = gaussian(I_hp, sigma=max(0.6, radius * 0.002))

    ring = disc_ring(disc_mask, radius)
    if np.any(fov_in & (~ring)):
        I_hp[ring] = np.median(I_hp[fov_in & (~ring)])

    v = frangi(
        I_hp,
        sigmas=np.arange(0.6, 6.0, 0.6),
        alpha=0.5,
        beta=0.5,
        gamma=15,
        black_ridges=False,
    )
    v[~fov_in] = 0

    vals = v[fov_in]
    v1, v99 = np.percentile(vals, [1, 99.5])
    v = np.clip((v - v1) / (v99 - v1 + 1e-8), 0, 1)
    v[~fov_in] = 0
    return v


def connected_hysteresis(
    v: np.ndarray,
    fov_in: np.ndarray,
    disc_mask: np.ndarray,
    bright_mask: np.ndarray,
    radius: float,
    hi_percentile: float,
    lo_percentile: float,
) -> np.ndarray:
    valid = fov_in & (~disc_mask)
    if np.sum(valid) < 1000:
        valid = fov_in

    hi = np.percentile(v[valid], hi_percentile)
    lo = np.percentile(v[valid], lo_percentile)

    no_go = morphology.dilation(bright_mask, disk(int(max(2, radius * 0.02))))
    candidates = ((v > lo) | disc_mask) & fov_in & (~no_go)

    ring_outer = morphology.dilation(disc_mask, disk(int(max(10, radius * 0.06))))
    ring_inner = morphology.dilation(disc_mask, disk(int(max(2, radius * 0.01))))
    ring = ring_outer & (~ring_inner)
    seeds = (v > hi) & ring & fov_in & (~no_go)

    if np.sum(seeds) < 20:
        seeds = (v > hi) & fov_in & (~no_go)
    seeds = seeds & candidates

    mask = morphology.reconstruction(
        seeds.astype(np.uint8), candidates.astype(np.uint8), method="dilation"
    ).astype(bool)
    mask = morphology.closing(mask, disk(2))
    return mask


def connected_hysteresis_with_stats(
    v: np.ndarray,
    fov_in: np.ndarray,
    disc_mask: np.ndarray,
    bright_mask: np.ndarray,
    radius: float,
    hi_percentile: float,
    lo_percentile: float,
) -> tuple[np.ndarray, dict]:
    valid = fov_in & (~disc_mask)
    if np.sum(valid) < 1000:
        valid = fov_in

    hi = float(np.percentile(v[valid], hi_percentile))
    lo = float(np.percentile(v[valid], lo_percentile))

    no_go = morphology.dilation(bright_mask, disk(int(max(2, radius * 0.02))))
    candidates = ((v > lo) | disc_mask) & fov_in & (~no_go)

    ring_outer = morphology.dilation(disc_mask, disk(int(max(10, radius * 0.06))))
    ring_inner = morphology.dilation(disc_mask, disk(int(max(2, radius * 0.01))))
    ring = ring_outer & (~ring_inner)
    seeds = (v > hi) & ring & fov_in & (~no_go)

    if np.sum(seeds) < 20:
        seeds = (v > hi) & fov_in & (~no_go)
    seeds = seeds & candidates

    mask = morphology.reconstruction(
        seeds.astype(np.uint8), candidates.astype(np.uint8), method="dilation"
    ).astype(bool)
    mask = morphology.closing(mask, disk(2))

    stats = {
        "seed_count": int(np.sum(seeds)),
        "candidate_count": int(np.sum(candidates)),
        "mask_fraction": float(np.mean(mask[fov_in])) if np.any(fov_in) else 0.0,
        "hi_percentile": float(hi_percentile),
        "lo_percentile": float(lo_percentile),
        "hi_threshold": hi,
        "lo_threshold": lo,
    }
    return mask, stats


def auto_select_hysteresis_mask(
    v: np.ndarray,
    fov_in: np.ndarray,
    disc_mask: np.ndarray,
    bright_mask: np.ndarray,
    radius: float,
    cfg: SegmentationConfig,
) -> tuple[np.ndarray, dict]:
    best: tuple[float, np.ndarray, dict] | None = None
    target_mid = 0.5 * (cfg.target_vessel_frac_min + cfg.target_vessel_frac_max)

    for hi_p in cfg.auto_hi_candidates:
        for lo_p in cfg.auto_lo_candidates:
            if lo_p >= hi_p:
                continue
            mask, stats = connected_hysteresis_with_stats(
                v=v,
                fov_in=fov_in,
                disc_mask=disc_mask,
                bright_mask=bright_mask,
                radius=radius,
                hi_percentile=hi_p,
                lo_percentile=lo_p,
            )

            frac = stats["mask_fraction"]
            seeds = stats["seed_count"]
            in_range = cfg.target_vessel_frac_min <= frac <= cfg.target_vessel_frac_max

            if in_range:
                score = abs(frac - target_mid)
            else:
                score = min(
                    abs(frac - cfg.target_vessel_frac_min),
                    abs(frac - cfg.target_vessel_frac_max),
                ) + 0.2

            if seeds < 20:
                score += 0.2
            if seeds < 10:
                score += 0.4

            if best is None or score < best[0]:
                best = (score, mask, stats)

    if best is None:
        mask = connected_hysteresis(
            v=v,
            fov_in=fov_in,
            disc_mask=disc_mask,
            bright_mask=bright_mask,
            radius=radius,
            hi_percentile=cfg.hi_percentile,
            lo_percentile=cfg.lo_percentile,
        )
        stats = {
            "seed_count": 0,
            "candidate_count": 0,
            "mask_fraction": float(np.mean(mask[fov_in])) if np.any(fov_in) else 0.0,
            "hi_percentile": float(cfg.hi_percentile),
            "lo_percentile": float(cfg.lo_percentile),
            "hi_threshold": float("nan"),
            "lo_threshold": float("nan"),
        }
        return mask, stats

    return best[1], best[2]


def remove_exudate_components(mask: np.ndarray, exu_mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    lab = measure.label(mask)
    exu_dil = morphology.dilation(exu_mask, disk(1))
    for region in measure.regionprops(lab):
        coords = region.coords
        exu_frac = float(np.mean(exu_dil[coords[:, 0], coords[:, 1]]))
        if exu_frac > 0.35 and region.area < 5000:
            out[lab == region.label] = False
    return out


def segment_vessel_mask(rgb: np.ndarray, cfg: SegmentationConfig) -> tuple[np.ndarray, dict]:
    fov, radius = compute_fov_and_radius(rgb, threshold=cfg.fov_threshold)
    fov_in = inner_fov_mask(fov, radius, cfg.inner_fov_fraction, cfg.inner_fov_min_band)

    g = to_float01(rgb[..., 1])
    disc_mask = detect_optic_disc(g, fov_in, radius, percentile=cfg.disc_percentile)
    exu_mask = detect_exudates(g, fov_in, radius, offset=cfg.exudate_offset)

    g_clean = inpaint_mask(g, exu_mask & fov_in, inpaint_radius=cfg.inpaint_radius)
    I = preprocess_green(np.dstack([g_clean, g_clean, g_clean]), fov_in, radius)
    v = vessel_probability(I, fov_in, radius, disc_mask)

    if cfg.use_auto_thresholds:
        mask, thr_info = auto_select_hysteresis_mask(
            v=v, fov_in=fov_in, disc_mask=disc_mask, bright_mask=exu_mask, radius=radius, cfg=cfg
        )
    else:
        mask, thr_info = connected_hysteresis_with_stats(
            v=v,
            fov_in=fov_in,
            disc_mask=disc_mask,
            bright_mask=exu_mask,
            radius=radius,
            hi_percentile=cfg.hi_percentile,
            lo_percentile=cfg.lo_percentile,
        )

    mask = remove_exudate_components(mask, exu_mask)
    mask = remove_small_components(mask, min_size=cfg.min_component_size)
    mask &= fov_in
    thr_info["final_mask_fraction"] = float(np.mean(mask[fov_in])) if np.any(fov_in) else 0.0
    return mask, thr_info


def save_mask_png(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_u8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_path), out_u8)


def iter_images(input_dir: Path, exts: Iterable[str]) -> list[Path]:
    exts_norm = {e.lower() for e in exts}
    return [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts_norm]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment retinal vessels into binary masks.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="Single input image path.")
    mode.add_argument("--input-dir", type=Path, help="Directory of input images.")

    parser.add_argument("--output", type=Path, help="Output mask path for single-image mode.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for batch mode.")
    parser.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"], help="Extensions for --input-dir.")

    parser.add_argument("--fixed-thresholds", action="store_true", help="Disable auto per-image threshold search.")
    parser.add_argument("--hi-percentile", type=float, default=95.0, help="Used when --fixed-thresholds is set.")
    parser.add_argument("--lo-percentile", type=float, default=82.0, help="Used when --fixed-thresholds is set.")
    parser.add_argument("--target-frac-min", type=float, default=0.08, help="Auto mode target min vessel fraction inside FOV.")
    parser.add_argument("--target-frac-max", type=float, default=0.25, help="Auto mode target max vessel fraction inside FOV.")
    parser.add_argument("--min-size", type=int, default=80)

    return parser.parse_args()


def run_single(input_path: Path, output_path: Path, cfg: SegmentationConfig) -> None:
    rgb = imread_rgb(input_path)
    mask, thr = segment_vessel_mask(rgb, cfg)
    save_mask_png(mask, output_path)
    print(f"Saved mask: {output_path}")
    print(
        f"Thresholds: hi_p={thr['hi_percentile']:.1f}, lo_p={thr['lo_percentile']:.1f}, "
        f"hi={thr['hi_threshold']:.4f}, lo={thr['lo_threshold']:.4f}, "
        f"mask_frac={thr['final_mask_fraction']:.4f}"
    )


def run_batch(input_dir: Path, output_dir: Path, cfg: SegmentationConfig, exts: Iterable[str]) -> None:
    images = iter_images(input_dir, exts)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(images)} images in {input_dir}")

    for idx, image_path in enumerate(images, start=1):
        rgb = imread_rgb(image_path)
        mask, thr = segment_vessel_mask(rgb, cfg)
        out_path = output_dir / f"{image_path.stem}_mask.png"
        save_mask_png(mask, out_path)
        print(
            f"[{idx}/{len(images)}] {image_path.name} -> {out_path.name} "
            f"(hi_p={thr['hi_percentile']:.1f}, lo_p={thr['lo_percentile']:.1f}, frac={thr['final_mask_fraction']:.4f})"
        )


def main() -> None:
    args = parse_args()
    cfg = SegmentationConfig(
        use_auto_thresholds=not args.fixed_thresholds,
        hi_percentile=args.hi_percentile,
        lo_percentile=args.lo_percentile,
        target_vessel_frac_min=args.target_frac_min,
        target_vessel_frac_max=args.target_frac_max,
        min_component_size=args.min_size,
    )

    if args.input is not None:
        if args.output is None:
            raise ValueError("Single-image mode requires --output.")
        run_single(args.input, args.output, cfg)
        return

    if args.output_dir is None:
        raise ValueError("Batch mode requires --output-dir.")
    run_batch(args.input_dir, args.output_dir, cfg, args.exts)


if __name__ == "__main__":
    main()
