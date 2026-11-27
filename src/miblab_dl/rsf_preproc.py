import logging

import numpy as np
import skimage.morphology


def rsf_candidates(fat, water, kidneys, fat_threshold=0.5, **kwargs):
    """Compute renal sinus fat mask candidates via the convex hull of the kidney mask.

    Within the convex hull of each kidney, voxels whose fat fraction
    (fat / (fat + water)) exceeds a threshold are labelled as sinus fat.

    Args:
        fat (np.ndarray): Fat-channel Dixon volume.
        water (np.ndarray): Water-channel Dixon volume, same shape as `fat`.
            Used together with `fat` to compute the fat fraction map.
        kidneys (np.ndarray): Kidney label mask (1=left kidney, 2=right kidney).
        fat_threshold (float): Voxels inside the convex hull with a fat
            fraction above this value are considered sinus fat candidates.
        **kwargs: Any keyword argument accepted by
            :func:`skimage.morphology.convex_hull_image`.

    Returns:
        np.ndarray: Label image of the same shape as `kidneys`, with sinus fat
        labelled 1 (left) and 2 (right).
    """
    rsf = np.zeros(kidneys.shape, dtype=np.uint8)
    fat_fraction = _fat_fraction_map(fat, water)

    for kidney in [1, 2]:
        mask = (kidneys == kidney)
        if not mask.any():
            continue

        try:
            kidney_hull = _convex_hull_image_3d(mask, **kwargs)
        except Exception as e:
            logging.error(f"Convex hull failed for label {kidney}: {e}")
            continue

        sinus_fat = kidney_hull & (fat_fraction > fat_threshold)
        rsf[sinus_fat] = kidney

    return rsf


def _fat_fraction_map(fat, water, eps=1e-6):
    """Fat fraction map: fat / (fat + water), clipped to [0, 1].

    eps prevents division by zero in background voxels where both fat
    and water are ~0.
    """
    fat = fat.astype(float)
    water = water.astype(float)
    denom = fat + water + eps
    return np.clip(fat / denom, 0.0, 1.0)


def _convex_hull_image_3d(array, **kwargs):
    """Convex hull computed slice by slice along the z-axis."""
    volume = np.around(array).astype(bool)
    hull = np.zeros(volume.shape, dtype=bool)
    for z in range(volume.shape[2]):
        slice_ = volume[:, :, z]
        if not slice_.any():
            continue
        hull[:, :, z] = skimage.morphology.convex_hull_image(slice_, **kwargs)
    return hull