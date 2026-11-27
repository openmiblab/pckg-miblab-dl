import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import median_filter
import vreg

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from miblab_data.zenodo import fetch as zenodo_fetch
from miblab_dl.cache import get_cache_dir



def _compute_fat_fraction(input_array):
    water = input_array[..., 2]
    fat = input_array[..., 3]

    # Safely compute fat fraction (avoids division by zero / NaN where water + fat == 0)
    denom = water + fat
    fat_fraction = np.divide(
        fat, denom, out=np.zeros_like(fat, dtype=np.float32), where=denom != 0
    )

    # clip fat fraction to [0, 1] range
    fat_fraction = np.clip(fat_fraction, 0.0, 1.0)

    # apply in-slice (last axis) median 3x3 filter
    # Assumes fat_fraction has shape (..., H, W, Z) where the last axis is slice (Z)
    # and the preceding two axes are the in-plane spatial dimensions (H, W)
    fat_fraction = median_filter(fat_fraction, size=(3, 3, 1))

    return fat_fraction


def _add_fat_fraction(input_array: np.ndarray):
    if input_array.shape[-1] == 5:
        return input_array
    
    fat_fraction = _compute_fat_fraction(input_array)

    # create output array which adds fat_fraction in the last dimension
    output_array = np.concatenate([input_array, fat_fraction[..., np.newaxis]], axis=-1)

    return output_array




def _add_fat_fraction_to_batch(batch_folder):
    """Create a 5-channel copy of a batch folder with fat fraction as channel 5.

    If every case in ``batch_folder`` already has 5 channels
    (``<case>_0000.nii.gz`` through ``<case>_0004.nii.gz``), no copy is made
    and ``batch_folder`` is returned unchanged.

    Otherwise, for each case with ``<case>_0000.nii.gz`` through
    ``<case>_0003.nii.gz`` in ``batch_folder``, the four channels are copied to
    a new sister folder and the fat fraction, computed from channels ``_0002``
    (water) and ``_0003`` (fat), is written there as ``<case>_0004.nii.gz``.
    The source folder is not changed.

    The new folder is named after the source folder with the suffix ``_5ch``
    (for example ``Dixon_batch`` becomes ``Dixon_batch_5ch``). If it already
    exists, it is deleted and created again, so it never contains files from
    earlier runs.

    Args:
        batch_folder (str | Path): Folder containing NIfTI files named
            ``<case>_0000.nii.gz`` through ``<case>_0003.nii.gz``.

    Returns:
        Path: Path to ``batch_folder`` if it already has 5 channels for all
            cases, otherwise the path to the new 5-channel folder.

    Raises:
        FileNotFoundError: If the folder has no ``*_0000.nii.gz`` files, or
            if a case is missing one of its four channel files.
    """
    batch_folder = Path(batch_folder).resolve()

    channel_0_files = sorted(batch_folder.glob("*_0000.nii.gz"))
    if not channel_0_files:
        raise FileNotFoundError(f"No '*_0000.nii.gz' files found in {batch_folder}")

    # Check all cases before creating anything
    cases = {}
    all_have_5_channels = True
    for ch0_path in channel_0_files:
        case = ch0_path.name[: -len("_0000.nii.gz")]
        channel_paths = []
        for i in range(4):
            src = batch_folder / f"{case}_{i:04d}.nii.gz"
            if not src.exists():
                raise FileNotFoundError(f"Missing expected channel file: {src}")
            channel_paths.append(src)
        cases[case] = channel_paths

        if not (batch_folder / f"{case}_0004.nii.gz").exists():
            all_have_5_channels = False

    if all_have_5_channels:
        return batch_folder

    # Sister folder, recreated from scratch
    out_folder = batch_folder.with_name(batch_folder.name + "_5ch")
    if out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True)

    for case, channel_paths in cases.items():

        # Copy the 4 original channels unchanged
        for src in channel_paths:
            shutil.copy2(src, out_folder / src.name)

        # Compute the fat fraction and write it as the 5th channel
        volumes = [vreg.read_nifti(p) for p in channel_paths]
        stacked = np.stack([v.values for v in volumes], axis=-1)

        fat_fraction = _compute_fat_fraction(stacked)
        ff_vol = vreg.volume(fat_fraction, volumes[0].affine)
        vreg.write_nifti(ff_vol, out_folder / f"{case}_0004.nii.gz")

    return out_folder




def _load_predictor(device=None, verbose=False):
    """Download the kidney model if needed and return an initialised predictor."""

    _MODEL = 'nnunet_rsf.zip'
    _MODEL_DOI = "24515900"

    if verbose:
        print('Downloading model..')

    weights_dir = get_cache_dir('rsf_dixon')
    weights_path = zenodo_fetch(_MODEL, weights_dir, _MODEL_DOI, extract=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device = torch.device(device_str)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=verbose,
    )
    predictor.initialize_from_trained_model_folder(
        os.path.join(weights_path, "nnUNetTrainer__nnUNetPlans__3d_fullres"),
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name='checkpoint_final.pth',
    )
    return predictor


def _to_nnunet(input_array, spacing):
    """Convert an [x, y, z, 5] array and its [x, y, z] spacing to nnU-Net's format.

    nnU-Net expects (channels, z, y, x) arrays and spacings in (z, y, x) order.
    """
    if input_array.ndim != 4 or input_array.shape[-1] != 5:
        raise ValueError(
            "input_array must have shape [x, y, z, 5] with out-phase, "
            f"in-phase, water, fat and fat fraction images; got {input_array.shape}."
        )
    spacing = [float(s) for s in spacing]
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError(f"spacing must be three positive numbers; got {spacing}.")

    data = np.ascontiguousarray(np.transpose(input_array, (3, 2, 1, 0)), dtype=np.float32)
    return data, {'spacing': tuple(spacing[::-1])}





def rsf_dixon(input_array, spacing, device=None, verbose=False):
    """Segment renal sinus fat on coronal Dixon images.

    This requires 4-channel coronal input data with out-phase images,
    in-phase images, water maps, and fat maps. 
    The function downloads the model weights from a 
    `Zenodo repository <https://zenodo.org/records/24515900>`_. 

    Args:
        input_array (numpy.ndarray): A 4D numpy array of shape
            [x, y, z, contrast] representing the input medical image
            volume. The last index must contain out-phase, in-phase,
            water and fat images, in that order.
        spacing (sequence of float): Voxel size in mm along
            the x, y and z axes of `input_array`. The model resamples
            the data to its own target spacing, so a wrong value
            gives wrong results. 
        device (str, optional): Processor on which to run the
            computation. Defaults to "cuda" if available and "cpu"
            otherwise.
        postproc (bool, optional): If True, select the largest connected
            component for each kidney mask. Defaults to True.
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.

    Returns:
        numpy.ndarray: Label image of the same shape as the first three
        axes of `input_array` (left kidney=1, right kidney=2).

    Raises:
        ValueError: If `input_array` or `spacing` has the wrong shape.
    """
    input_array = _add_fat_fraction(input_array)
    data, properties = _to_nnunet(input_array, spacing)
    predictor = _load_predictor(device, verbose)

    if verbose:
        print('Applying model to data..')

    segmentation = predictor.predict_single_npy_array(data, properties)

    # Back to (x, y, z)
    kidneys = np.transpose(segmentation, (2, 1, 0))

    return kidneys


def rsf_dixon_batch(input_folder, output_folder, device=None,
                          verbose=False,
                          num_processes_preprocessing=3, num_processes_export=3):
    """Segment renal sinus fat on coronal Dixon images, for many cases on disk.

    The input folder holds one set of four NIfTI files per case, named
    ``<case>_0000.nii.gz`` (out-phase), ``<case>_0001.nii.gz`` (in-phase),
    ``<case>_0002.nii.gz`` (water) and ``<case>_0003.nii.gz`` (fat). A label
    image ``<case>.nii.gz`` is written to the output folder for each case
    (left kidney=1, right kidney=2). 
    The function downloads the model weights from a 
    `Zenodo repository <https://zenodo.org/records/24515900>`_. 

    Preprocessing and export of the results run in parallel worker processes
    while the network predicts, so this is much faster than calling
    :func:`kidney_dixon` in a loop. On Windows, call it from a script
    inside an ``if __name__ == "__main__":`` block.

    Args:
        input_folder (str or Path): Folder with the input NIfTI files.
        output_folder (str or Path): Folder for the label images. It is
            created if it does not exist, and existing files with the same
            name are overwritten.
        device (str, optional): Processor on which to run the computation.
            Defaults to "cuda" if available and "cpu" otherwise.
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.
        num_processes_preprocessing (int, optional): Number of worker
            processes for preprocessing. Defaults to 3.
        num_processes_export (int, optional): Number of worker processes
            for resampling and exporting the predictions. Defaults to 3.

    Raises:
        FileNotFoundError: If the input folder does not exist.
        ValueError: If the input folder contains no ``*_0000.nii.gz`` files.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    input_folder_ff = _add_fat_fraction_to_batch(input_folder)
    predictor = _load_predictor(device, verbose)

    if verbose:
        print('Applying model to data..')

    # nnU-Net writes raw labels plus some json files, so predict into a
    # temporary folder and only copy the final label images to the output folder.
    with tempfile.TemporaryDirectory(prefix='tmp_', dir=get_cache_dir()) as temp_dir:

        # nnU-Net only accepts plain strings as folder arguments
        predictor.predict_from_files(
            str(input_folder_ff),
            temp_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_export,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

        for result in sorted(Path(temp_dir).glob('*.nii.gz')):
            nii = nib.load(result)
            labels = np.asanyarray(nii.dataobj)

            # Keep the geometry of the nnU-Net output
            out = nib.Nifti1Image(labels, nii.affine, nii.header)
            out.set_data_dtype(np.int16)
            nib.save(out, output_folder / result.name)

    # Remove computed database
    if input_folder_ff != input_folder:
        shutil.rmtree(input_folder_ff)
