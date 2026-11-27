import numpy as np
import dbdicom as db
import vreg

from miblab_dl.fetch_data import fetch
from miblab_dl.cache import get_cache_dir

def load_dixon_case(case='Dixon'):
    """Load a Dixon test case from DICOM as a 4D array.

    The four channels are read in the order expected by the Dixon models:
    out-phase, in-phase, water and fat.

    Args:
        case (str, optional): Name of the test case to fetch.
            Defaults to 'Dixon'.

    Returns:
        tuple: A tuple (array, spacing, affine) where

        - **array** (np.ndarray): 4D array of shape (x, y, z, 4) with the
          channels stacked along the last axis.
        - **spacing**: Voxel spacing of the volumes, taken from the
          out-phase volume and assumed to be identical for all channels.
        - **affine**: Affine array in LPS+ coordinates (DICOM, ITK)
    """
    folder = fetch(case)

    # Read DICOM data in the correct order. 
    desc = ['out_phase', 'in_phase', 'water', 'fat']
    dicom_series = [db.series(folder, contains=d)[0] for d in desc]
    volumes = [db.volume(series, verbose=0) for series in dicom_series]

    # Convert to a 4D input array and read spacing
    array = np.stack([v.values for v in volumes], axis=-1)
    spacing = volumes[0].spacing
    affine = volumes[0].affine

    return array, spacing, affine

def load_dixon_case_as_batch(case='Dixon'):
    """Write the Dixon test case as NIfTI files for kidney_dixon_batch.

    The four channels are saved as ``<case>_0000.nii.gz`` (out-phase),
    ``<case>_0001.nii.gz`` (in-phase), ``<case>_0002.nii.gz`` (water) and
    ``<case>_0003.nii.gz`` (fat) in a cache folder named ``Dixon_batch``.

    Note:
        The files are currently written with an identity affine, so the
        voxel spacing and orientation are **not** preserved. This needs to
        be revisited once the model is adjusted.

    Args:
        case (str, optional): Name of the test case to fetch, also used as
            the prefix of the output files. Defaults to 'Dixon'.

    Returns:
        Path: The folder with the NIfTI files.
    """
    folder = fetch(case)

    # Read DICOM data in the correct order
    desc = ['out_phase', 'in_phase', 'water', 'fat']
    dicom_series = [db.series(folder, contains=d)[0] for d in desc]
    volumes = [db.volume(series, verbose=0) for series in dicom_series]

    batch_folder = get_cache_dir('Dixon_batch')
    for i, volume in enumerate(volumes):
        nifti_file = batch_folder / f'{case}_{i:04d}.nii.gz'
        vreg.write_nifti(volume, nifti_file)

    return batch_folder