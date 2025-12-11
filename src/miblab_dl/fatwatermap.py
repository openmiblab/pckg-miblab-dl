"""
Compute water-dominance masks from data that have fat and water maps
"""

import os
import sys
import subprocess
import shutil
from platformdirs import user_cache_dir
from pathlib import Path


import numpy as np
import nibabel as nib
from miblab import zenodo_fetch




def cleanup():
    cachedir = Path(user_cache_dir("miblab-dl"))
    shutil.rmtree(cachedir)


def _remake_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _cache_dir(cache=None):

    # 1. User override via environment variable
    if cache:
        try:
            os.makedirs(cache, exist_ok=True)
        except Exception:
            # If user has set an invalid/unwritable path, raise an error
            raise ValueError(
                f"{cache} is not a valid cache directory for miblab-dl."
            )
        else:
            return cache

    # 2. Fallback to platform-specific user cache (~/.cache/miblab-dl)
    cache_dir = user_cache_dir("miblab-dl")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def fatwater(op_phase, in_phase, te_o=None, te_i=None, t2s_w=15, t2s_f=10, cache=None):
    """Compute fat and water maps from opposed-phase and in-phase arrays

    Args:
        op_phase (np.ndarray): opposed phase data
        in_phase (np.ndarray): in-phase data
        model (str): path to the model files
        cache (str, optional): directory to use for storing model weights and temp files
           This defaults to the standard cache dir location of the operating system.

    Returns:
        fat, water: numpy arrays of the same shape and type as the input arrays.
    """
    print('Downloading model..')

    # Persistent cache memory for storing model weights avoids the 
    # need to download every time.
    cachedir = _cache_dir(cache)
    model = zenodo_fetch("FatWaterPredictor.zip", cachedir, "17791059", extract=True)
    
    print('Predicting fat and water images..')

    waterdom = _predict_mask_numpy(model, op_phase, in_phase, cachedir)
    fat, water = _compute_fatwater(waterdom, op_phase, in_phase, te_o, te_i, t2s_w, t2s_f)
    fat[fat < 0] = 0
    water[water < 0] = 0
    return fat, water



def _predict_mask_numpy(model, op_phase, in_phase, cachedir):
    
    # Making temporary folders in persistent cache is safer than tempfile on HPC
    input_folder = os.path.join(cachedir, 'input_folder')
    output_folder = os.path.join(cachedir, 'output_folder')
    _remake_dir(input_folder)
    _remake_dir(output_folder)

    # Save numpy arrays as nifti
    case_id = "dixon"
    file_op = os.path.join(input_folder, f"{case_id}_0000.nii.gz")
    file_ip = os.path.join(input_folder, f"{case_id}_0001.nii.gz")
    nifti_op = nib.Nifti1Image(op_phase, np.eye(4))
    nifti_ip = nib.Nifti1Image(in_phase, np.eye(4))
    nib.save(nifti_op, file_op)
    nib.save(nifti_ip, file_ip)

    # Create predictions in a temporary output_folder
    _predict_mask_folder(model, input_folder, output_folder, cachedir)

    # Return result as binary numpy array
    mask_file = os.path.join(output_folder, f"{case_id}.nii.gz")
    waterdom = nib.load(mask_file).get_fdata().astype(np.int8)
    
    # Clean up temp dirs
    shutil.rmtree(input_folder)
    shutil.rmtree(output_folder)  

    return waterdom


def _predict_mask_folder(model, input_folder, output_folder, cachedir):

    # These two variables are not used but we are setting to a 
    # dummy value to silence the warnings
    os.environ["nnUNet_raw"] = os.getcwd() 
    os.environ["nnUNet_preprocessed"] = os.getcwd()

    # Folder containing the model weights
    os.environ["nnUNet_results"] = model

    # Making temporary folders in persistent cache
    predictions = os.path.join(cachedir, 'predictions')
    _remake_dir(predictions)
    
    # Predict and save results in the temporary folder
    cmd = [
        "nnUNetv2_predict",
        "-d", "Dataset001_FatWaterPredictor",
        "-i", input_folder,
        "-o", predictions,
        "-f", "0", "1", "2", "3", "4",
        "-tr", "nnUNetTrainer",
        "-c", "3d_fullres",
        "-p", "nnUNetPlans",
    ]
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        encoding="utf-8",   # <-- force UTF-8 decoding
        errors="replace"    # <-- avoids crash if weird bytes appear
    )

    # Stream logs in real-time
    for line in process.stdout:
        print(line, end="")

    process.wait()  # wait for completion
    
    # Run post-processing
    os.makedirs(output_folder, exist_ok=True)
    source = os.path.join(model, 'Dataset001_FatWaterPredictor', 'nnUNetTrainer__nnUNetPlans__3d_fullres', "crossval_results_folds_0_1_2_3_4")
    pproc = os.path.join(source, 'postprocessing.pkl')
    plans = os.path.join(source, 'plans.json')

    cmd = [
        "nnUNetv2_apply_postprocessing",
        "-i", predictions,
        "-o", output_folder,
        "-pp_pkl_file", pproc,
        "-np", "8",
        "-plans_json", plans,
    ]

    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        encoding="utf-8",   # <-- force UTF-8 decoding
        errors="replace"    # <-- avoids crash if weird bytes appear
    )

    # Stream logs in real-time
    for line in process.stdout:
        print(line, end="")

    process.wait()  # wait for completion

    shutil.rmtree(predictions)


def _compute_fatwater(waterdom, op_phase, in_phase, te_o, te_i, t2s_w, t2s_f):

    if te_o is None:
        Eof, Eif, Eow, Eiw = 1, 1, 1, 1
    else:
        Eof = np.exp(-te_o/t2s_f)
        Eif = np.exp(-te_i/t2s_f)
        Eow = np.exp(-te_o/t2s_w)
        Eiw = np.exp(-te_i/t2s_w)

    Efatdom = np.array([[Eof, -Eow], [Eif, Eiw]])
    Ewatdom = np.array([[-Eof, Eow], [Eif, Eiw]])

    Efatdom_inv = np.linalg.inv(Efatdom)
    Ewatdom_inv = np.linalg.inv(Ewatdom)

    fat, water = _apply_pixelwise_matrix(op_phase, in_phase, waterdom, Efatdom_inv, Ewatdom_inv)

    return fat, water


def _apply_pixelwise_matrix(a, b, mask, M0, M1) -> np.ndarray:
    """
    For each pixel/voxel combine [a, b] as a 2-vector v and compute:
        result = M0 @ v   if mask == 0/False
        result = M1 @ v   if mask == 1/True
    Returns arrays c, d of same shape and type

    Parameters
    ----------
    a, b : np.ndarray
        Input 3D arrays of the same shape (spatial).
    mask : np.ndarray
        Boolean/0-1 array same shape as `a`/`b`. True selects M1.
    M0, M1 : array-like (2x2)
        Two 2x2 matrices.

    Returns
    -------
    np.ndarray
        Output 3D arrays of the same shape (spatial).
    """

    # stack components into last axis: shape (..., 2)
    v = np.stack((a, b), axis=-1).astype(float)  # shape (Z, Y, X, 2)

    # compute results for both matrices: result = v @ M.T  (vector @ M.T => M @ v per-voxel)
    res0 = v @ M0.T   # shape (..., 2)
    res1 = v @ M1.T

    # Select based on mask; expand mask to last axis
    mask_bool = np.asarray(mask, dtype=bool)
    mask_expanded = mask_bool[..., None]   # shape (..., 1)

    result = np.where(mask_expanded, res1, res0)  # shape (..., 2)

    return result[...,0], result[...,1]

