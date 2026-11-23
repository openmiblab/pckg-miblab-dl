import os
import tempfile

from tqdm import tqdm
import numpy as np
import vreg

from totalsegmentator import map_to_binary
from totalsegmentator.python_api import totalsegmentator

from miblab_dl.cache import get_cache_dir


def _totseg(vol, cutoff=None, task='total', roi_subset=None, **kwargs):

    # Creating tmp files in persistent cache is safer on HPC
    with tempfile.TemporaryDirectory(prefix="tmp_", dir=get_cache_dir()) as temp_dir:
        print('Saving source as nifti..')
        nifti_file = os.path.join(temp_dir, 'source.nii.gz')
        vreg.write_nifti(vol, nifti_file)

        print('Segmenting organs..')
        totalsegmentator(nifti_file,  temp_dir, task=task, roi_subset=roi_subset, **kwargs)

        if roi_subset is None:
            roi_set = list(map_to_binary.class_map[task].values())
        else:
            roi_set = roi_subset

        mask = {}
        for roi in tqdm(roi_set, desc='Reading results..'):
            roifile = os.path.join(temp_dir, roi + '.nii.gz')
            v = vreg.read_nifti(roifile)
            if cutoff is not None:
                values = v.values
                values[values > cutoff] = 1
                values[values <= cutoff] = 0
                v.set_values(values)
            mask[roi] = v

    return mask


def totseg(vol, cutoff=None, **kwargs): 
    """Run totalsegmentator on one or more volumes.

    Source: `totalsegmentator <https://github.com/wasserth/TotalSegmentator>`_.

    Args:
        vol (vreg.Volume3D or list): Either a single volume, or a list 
            of volumes to be segmented.
        cutoff (float, optional): Pixels with a probability higher 
            than cutoff will be included in the mask. If cutoff is 
            not provided, probabilities will be returned directly. 
            Defaults to None.
        kwargs: Any keyword arguments accepted by the `totalsegmentor 
            python API <https://github.com/wasserth/TotalSegmentator/tree/master?tab=readme-ov-file#totalsegmentator>`_.

    Returns:
        vreg.Volume3D: 
            A vreg volume with the label image. See 
            `totalsegmentator documentation <https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py>`_ 
            for a dictionary mapping labels to anatomical structures.

    Example:
        Use a machine with a cpu to run the task 'total_mr' on a 
        single volume saved as a nifti file:

        >>> import miblab
        >>> import vreg
        >>> vol = vreg.read_nifti('path/to/volume.nii.gz')
        >>> label = miblab.totseg(vol, cutoff=0.01, task='total_mr', device='cpu')
        >>> vreg.write_nifti(label, 'path/to/label.nii.gz')
    """

    if not isinstance(vol, list):
        total = _totseg(vol, cutoff, **kwargs)
        # Convert to label
        label_img = np.zeros(vol.shape, dtype=np.int16)
        for j, roi in enumerate(total):
            label_img += (j+1) * total[roi].values.astype(np.int16)
        return vreg.volume(label_img, vol.affine)

    total = {}
    for v in tqdm(vol, desc='Segmenting volumes..'):
        mask = _totseg(v, **kwargs)
        for roi in mask:
            if roi in total:
                values = total[roi].values + mask[roi].values
                total[roi].set_values(values)
            else:
                total[roi] = mask[roi]

    for roi, v in tqdm(total.items(), desc='Combining results..'):
        values = v.values/len(vol)
        if cutoff is not None:
            values[values > cutoff] = 1
            values[values <= cutoff] = 0
        v.set_values(values)
        total[roi] = v

    # Convert to label
    label_img = np.zeros(vol[0].shape, dtype=np.int16)
    for j, roi in enumerate(total):
        label_img += (j+1) * total[roi].values.astype(np.int16)

    # Return volume
    return vreg.volume(label_img, vol[0].affine)