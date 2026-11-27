import os
import tempfile
from pathlib import Path
import re
import shutil

import numpy as np
import vreg
import torch
import nibabel as nib
import scipy.ndimage as ndi

from miblab_data.zenodo import fetch as zenodo_fetch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from miblab_dl.cache import get_cache_dir



def _load_predictor(device=None, verbose=False):
    """Download the kidney model if needed and return an initialised predictor."""

    _MODEL = 'nnunet_kidneys_v3.zip'
    _MODEL_DOI = "24545951"

    if verbose:
        print('Downloading model..')

    weights_dir = get_cache_dir('kidney_dixon')
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
        use_mirroring=True,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=verbose,
    )
    predictor.initialize_from_trained_model_folder(
        os.path.join(weights_path, "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"),
        use_folds=("all",),
        checkpoint_name='checkpoint_best.pth',
    )
    return predictor


def _to_nnunet(input_array, spacing):
    """Convert an [x, y, z, 4] array and its [x, y, z] spacing to nnU-Net's format.

    nnU-Net expects (channels, z, y, x) arrays and spacings in (z, y, x) order.
    """
    if input_array.ndim != 4 or input_array.shape[-1] != 4:
        raise ValueError(
            "input_array must have shape [x, y, z, 4] with out-phase, "
            f"in-phase, water and fat images; got {input_array.shape}."
        )
    spacing = [float(s) for s in spacing]
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError(f"spacing must be three positive numbers; got {spacing}.")

    data = np.ascontiguousarray(np.transpose(input_array, (3, 2, 1, 0)), dtype=np.float32)
    return data, {'spacing': tuple(spacing[::-1])}


def kidney_dixon(input_array, spacing, device=None, verbose=False):
    """Segment individual kidneys on coronal Dixon images.

    This requires 4-channel coronal input data with out-phase images,
    in-phase images, water maps, and fat maps.

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
        clear_cache (bool, optional): If True, remove the downloaded
            model weights after the run. Defaults to False.
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.

    Returns:
        numpy.ndarray: Label image of the same shape as the first three
        axes of `input_array` (left kidney=1, right kidney=2).

    Raises:
        ValueError: If `input_array` or `spacing` has the wrong shape.
    """
    # The current model is trained with data incorrectly saved with spacing (1, 1, 1).
    # In order to correct the error, the spacing must be set to this value at 
    # inference as well. TODO: Retrain and remove this
    spacing = (1.0, 1.0, 1.0)

    # Model is trained with out-phase, in-phase, fat and water, 
    # But the package convention uses water and fat, so needs to be swapped.
    out = input_array.copy()
    out[..., [-2, -1]] = out[..., [-1, -2]]

    data, properties = _to_nnunet(out, spacing)

    predictor = _load_predictor(device, verbose)

    if verbose:
        print('Applying model to data..')

    segmentation = predictor.predict_single_npy_array(data, properties)

    # Back to (x, y, z)
    kidneys = _label_kidneys(np.transpose(segmentation, (2, 1, 0)))

    return kidneys


def _swap_fat_water(src):
    a=2
    b=3
    src = Path(src)
    dst = src.with_name(src.name + "_swap")
    dst.mkdir(parents=True, exist_ok=True)

    swap = {f"{a:04d}": f"{b:04d}", f"{b:04d}": f"{a:04d}"}
    pattern = re.compile(r"^(?P<case>.+)_(?P<ch>\d{4})\.nii\.gz$")

    for f in sorted(src.glob("*.nii.gz")):
        m = pattern.match(f.name)
        if m is None:
            print(f"Skipping (unexpected name): {f.name}")
            continue
        ch = swap.get(m["ch"], m["ch"])
        shutil.copy2(f, dst / f"{m['case']}_{ch}.nii.gz")

    return dst


def _set_affine(src: str | Path, affine: np.ndarray) -> Path:
    src_path = Path(src)
    dst = src_path.parent / f"{src_path.name}_isotropic"
    dst.mkdir(parents=True, exist_ok=True)

    for file_path in src_path.iterdir():
        if file_path.is_file() and (
            file_path.name.endswith(".nii") or file_path.name.endswith(".nii.gz")
        ):
            vol = vreg.read_nifti(file_path)
            new_vol = vreg.volume(vol.values, affine)
            vreg.write_nifti(new_vol, dst / file_path.name)

    return dst



def kidney_dixon_batch(input_folder, output_folder, device=None,
                          verbose=False,
                          num_processes_preprocessing=3, num_processes_export=3):
    """Segment individual kidneys on coronal Dixon images, for many cases on disk.

    The input folder holds one set of four NIfTI files per case, named
    ``<case>_0000.nii.gz`` (out-phase), ``<case>_0001.nii.gz`` (in-phase),
    ``<case>_0002.nii.gz`` (water) and ``<case>_0003.nii.gz`` (fat). A label
    image ``<case>.nii.gz`` is written to the output folder for each case
    (left kidney=1, right kidney=2).

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

    # The current model is trained with data incorrectly saved with spacing (1, 1, 1).
    # In order to correct the error, the spacing must be set to this value at 
    # inference as well. TODO: Retrain and remove this
    input_folder_affine = _set_affine(input_folder, np.eye(4))

    input_folder_swap = _swap_fat_water(input_folder_affine)
    predictor = _load_predictor(device, verbose)

    if verbose:
        print('Applying model to data..')

    # nnU-Net writes raw labels plus some json files, so predict into a
    # temporary folder and only copy the final label images to the output folder.
    with tempfile.TemporaryDirectory(prefix='tmp_', dir=get_cache_dir()) as temp_dir:

        # nnU-Net only accepts plain strings as folder arguments
        predictor.predict_from_files(
            str(input_folder_swap),
            temp_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_export,
        )

        for result in sorted(Path(temp_dir).glob('*.nii.gz')):
            nii = nib.load(result)
            labels = _label_kidneys(np.asanyarray(nii.dataobj))

            # Keep the geometry of the nnU-Net output
            out = nib.Nifti1Image(labels, nii.affine, nii.header)
            out.set_data_dtype(np.int16)
            nib.save(out, output_folder / result.name)

    # Remove temporary
    shutil.rmtree(input_folder_affine)
    shutil.rmtree(input_folder_swap)


def _label_kidneys(output_array):
    """Convert raw nnU-Net labels to a label image (left=1, right=2)."""

    left_kidney, right_kidney = _kidney_masks(output_array)
    return (left_kidney + 2*right_kidney).astype(np.int16)


def _largest_cluster(array:np.ndarray)->np.ndarray:
    """Given a mask array, return a new mask array containing only the largesr cluster.

    Args:
        array (np.ndarray): mask array with values 1 (inside) or 0 (outside)

    Returns:
        np.ndarray: mask array with only a single connect cluster of pixels.
    """
    # Label all features in the array
    label_img, cnt = ndi.label(array)
    # Find the label of the largest feature
    labels = range(1,cnt+1)
    size = [np.count_nonzero(label_img==l) for l in labels]
    max_label = labels[size.index(np.amax(size))]
    # Return a mask corresponding to the largest feature
    return label_img==max_label

def _kidney_masks(output_array:np.ndarray)->tuple:
    """Extract kidney masks from the output array of the UNETR

    Args:
        output_array (np.ndarray): 3D numpy array (x,y,z) with integer labels (0=background, 1=right kidney, 2=left kidney)

    Returns:
        tuple: A tuple of 3D numpy arrays (left_kidney, right_kidney) with masks for the kidneys.
    """
    left_kidney = _largest_cluster(output_array == 1)
    right_kidney = _largest_cluster(output_array == 2)

    return left_kidney, right_kidney