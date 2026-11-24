import os
import tempfile
from pathlib import Path

import numpy as np
import vreg
import torch
import nibabel as nib
import scipy.ndimage as ndi

from miblab_data.zenodo import fetch as zenodo_fetch
from monai.networks.nets.unetr import UNETR
from monai.inferers import sliding_window_inference
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from miblab_dl.cache import get_cache_dir
from miblab_dl.cache import clear_cache as clear_cache_function



def kidney_dixon_unetr(input_vol, device=None, overlap=0.3, postproc=True, clear_cache = False, verbose=False):

    """
    Segment individual kidneys on post-contrast Dixon images.

    This requires 4-channel input data with out-phase images, 
    in-phase images, water maps, and fat maps.

    Args:
        input_vol (vreg.Volume3D): A 4D vreg volume of shape 
            [x, y, z, contrast] representing the input medical image 
            volume. The last index must contain out-phase, in-phase, 
            water and fat images, in that order.
        device (str):
            processor on which to deply the computation. If this is not 
            provided, this defaults to "cuda" (if this is available) and 
            "cpu" otherwise.
        overlap (float): only valid for model = 'unetr' defines the amount of overlap between 
            adjacent sliding window patches during inference. A 
            higher value (e.g., 0.5) improves prediction smoothness 
            at patch borders but increases computation time.
        postproc (bool): If True, applies post-processing to select 
            the largest connected component from the UNETR output 
            for each kidney mask
        clear_cache: If True, the downloaded pth file is removed 
            again after running the inference.
        verbose (bool): If True, prints logging messages.

    Returns:
        vreg.Volume3D: 
            A volume with a kidneys as label array (left=1, right=2).
    """


    MODEL = 'UNETR_kidneys_v2.pth'
    MODEL_DOI = "15521814"

    input_array = input_vol.values

    # Format input array
    # ------------------
    # TODO: make this consistent for future training
    # swap the last two indices because UNETR is trained 
    # with opposed-phase/in-phase/fat/water
    input_array = input_array[..., [0, 1, 3, 2]]

    # flip
    if not input_vol.is_right_handed:
        input_array = np.flip(input_array, axis=2)

    # Rescale to required matrix size
    old_shape = input_array.shape
    new_shape = (320, 320, 144, 4)
    zoom_factors = [n / o for n, o in zip(new_shape, old_shape)]
    input_array = ndi.zoom(input_array, zoom_factors, order=1) 

    # from (x,y,z,c) to (c,y,x,z)
    input_array = np.transpose(input_array, (3, 0, 1, 2)) 

    # Normalize data
    input_array_out   = (input_array[0,...]-np.average(input_array[0,...]))/np.std(input_array[0,...])
    input_array_in    = (input_array[1,...]-np.average(input_array[1,...]))/np.std(input_array[1,...])
    input_array_water = (input_array[2,...]-np.average(input_array[2,...]))/np.std(input_array[2,...])
    input_array_fat   = (input_array[3,...]-np.average(input_array[3,...]))/np.std(input_array[3,...])

    input_array = np.stack((input_array_out, input_array_in, input_array_water, input_array_fat), axis=0)
    # Convert to NCHW[D] format: (1,c,y,x,z)
    # NCHW[D] stands for: batch N, channels C, height H, width W, depth D
    input_array = input_array.transpose(0,2,1,3) # from (x,y,z) to (y,x,z)
    input_array = np.expand_dims(input_array, axis=(0))
    input_array = input_array.astype(np.float32)

    if verbose:
        print('(Down)loading model..')

    weights_dir = get_cache_dir('kidney_dixon_unetr')
    weights_path = zenodo_fetch(MODEL, weights_dir, MODEL_DOI)

    if verbose:
        print('Applying model to data..')

    # Setup device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device = torch.device(device_str)

    # Define model architecture
    model = UNETR(
        in_channels=4,
        out_channels=3, # BACKGROUND, RIGHT KIDNEY (left on image), LEFT KIDNEY (right on image)
        img_size=(80, 80, 80),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
    model.to(device)

    # Convert to tensor
    input_tensor = torch.tensor(input_array).to(device)

    # Load model weights
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    #model.load_state_dict(weights)

    result = model.load_state_dict(weights, strict=False)

    # The model was trained in MONAI 1.3.2. These lines added for compatibility with MONAI 1.6.2
    # Tolerate only the unused cross-attention norms added in newer MONAI versions
    missing = [k for k in result.missing_keys if "norm_cross_attn" not in k]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected mismatch loading weights. Missing: {missing}, "
            f"Unexpected: {result.unexpected_keys}"
        )
    
    model.eval() 

    with torch.no_grad():
        output_tensor = sliding_window_inference(input_tensor, (80,80,80), 4, model, overlap=overlap, device=device_str, progress=True) 

    if verbose:
        print('Post-processing results...')

    # From probabilities for each channel to label image
    output_tensor = torch.argmax(output_tensor, dim=1)

    # Convert to numpy
    output_array = output_tensor.numpy(force=True)[0,:,:,:]
        
    # Transpose to original shape
    output_array = output_array.transpose(1,0,2) #from (y,x,z) to (x,y,z)

    # Rescale to original matrix size
    zoom_factors = [n / o for n, o in zip(old_shape[:3], new_shape[:3])]
    output_array = ndi.zoom(output_array, zoom_factors, order=1) 

    # flip back
    if not input_vol.is_right_handed:
        output_array = np.flip(output_array, axis=2)
    
    if postproc == True:
        left_kidney, right_kidney = _kidney_masks(output_array)
    else:
        left_kidney=output_array[output_array == 2]
        left_kidney[left_kidney==2]=1
        right_kidney=output_array[output_array == 1]

    kidneys = (left_kidney + 2*right_kidney).astype(np.int16)

    if clear_cache:
        if verbose:
            print('Deleting downloaded files...')
        clear_cache_function('kidney_dixon_unetr')

    return vreg.volume(kidneys, input_vol.affine) 



def _load_predictor(device=None, verbose=False):
    """Download the kidney model if needed and return an initialised predictor."""

    _MODEL = 'nnunet_kidneys_v2.zip'
    _MODEL_DOI = "15328218"

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
        tile_step_size=0.75,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device_str == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=verbose,
    )
    predictor.initialize_from_trained_model_folder(
        os.path.join(weights_path, "nnUNetTrainer__nnUNetPlans__3d_fullres"),
        use_folds='all',
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


def kidney_dixon(input_array, spacing=(1.0, 1.0, 1.0), device=None, postproc=True,
                    clear_cache=False, verbose=False):
    """Segment individual kidneys on post-contrast Dixon images.

    This requires 4-channel input data with out-phase images,
    in-phase images, water maps, and fat maps.

    Args:
        input_array (numpy.ndarray): A 4D numpy array of shape
            [x, y, z, contrast] representing the input medical image
            volume. The last index must contain out-phase, in-phase,
            water and fat images, in that order.
        spacing (sequence of float, optional): Voxel size in mm along
            the x, y and z axes of `input_array`. The model resamples
            the data to its own target spacing, so a wrong value
            gives wrong results. Defaults to (1.0, 1.0, 1.0).
        device (str, optional): Processor on which to run the
            computation. Defaults to "cuda" if available and "cpu"
            otherwise.
        postproc (bool, optional): If True, select the largest connected
            component for each kidney mask. Defaults to True.
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
    data, properties = _to_nnunet(input_array, spacing)

    predictor = _load_predictor(device, verbose)

    if verbose:
        print('Applying model to data..')

    segmentation = predictor.predict_single_npy_array(data, properties)

    # Back to (x, y, z)
    kidneys = _label_kidneys(np.transpose(segmentation, (2, 1, 0)), postproc)

    if clear_cache:
        if verbose:
            print('Deleting downloaded files...')
        clear_cache_function('kidney_dixon')

    return kidneys


def kidney_dixon_batch(input_folder, output_folder, device=None, postproc=True,
                          clear_cache=False, verbose=False,
                          num_processes_preprocessing=3, num_processes_export=3):
    """Segment individual kidneys on post-contrast Dixon images, for many cases on disk.

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
        postproc (bool, optional): If True, select the largest connected
            component for each kidney mask. Defaults to True.
        clear_cache (bool, optional): If True, remove the downloaded model
            weights after the run. Defaults to False.
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
    predictor = _load_predictor(device, verbose)

    if verbose:
        print('Applying model to data..')

    # nnU-Net writes raw labels plus some json files, so predict into a
    # temporary folder and only copy the final label images to the output folder.
    with tempfile.TemporaryDirectory(prefix='tmp_', dir=get_cache_dir()) as temp_dir:

        # nnU-Net only accepts plain strings as folder arguments
        predictor.predict_from_files(
            str(input_folder),
            temp_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_export,
        )

        for result in sorted(Path(temp_dir).glob('*.nii.gz')):
            nii = nib.load(result)
            labels = _label_kidneys(np.asanyarray(nii.dataobj), postproc)

            # Keep the geometry of the nnU-Net output
            out = nib.Nifti1Image(labels, nii.affine, nii.header)
            out.set_data_dtype(np.int16)
            nib.save(out, output_folder / result.name)

    if clear_cache:
        if verbose:
            print('Deleting downloaded files...')
        clear_cache_function('kidney_dixon')


def _label_kidneys(output_array, postproc):
    """Convert raw nnU-Net labels to a label image (left=1, right=2)."""
    if postproc:
        left_kidney, right_kidney = _kidney_masks(output_array)
    else:
        left_kidney = (output_array == 2).astype(np.int16)
        right_kidney = (output_array == 1).astype(np.int16)
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
    left_kidney = _largest_cluster(output_array == 2)
    right_kidney = _largest_cluster(output_array == 1)

    return left_kidney, right_kidney