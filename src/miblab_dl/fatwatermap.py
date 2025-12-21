"""
Compute water-dominance masks from data that have fat and water maps
"""

import os
import sys
import subprocess
import shutil
from platformdirs import user_cache_dir
from pathlib import Path
import tempfile

import numpy as np
import nibabel as nib
from miblab import zenodo_fetch
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder



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


def clear_cache(cache=None):
    cachedir = _cache_dir(cache)
    shutil.rmtree(cachedir)



def fatwater(
        op_phase, in_phase, 
        te_o=None, te_i=None, t2s_w=30, t2s_f=15, 
        tr=None, fa=None, t1_w=1400, t1_f=350, 
        cache=None,
    ):
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

    # Making temporary folders in persistent cache is safer on HPC
    tmp = tempfile.mkdtemp(prefix="tmp_", dir=cachedir)

    # Compute
    waterdom = _predict_mask_numpy(model, op_phase, in_phase, tmp)
    fat, water = _compute_fatwater(waterdom, op_phase, in_phase, te_o, te_i, t2s_w, t2s_f, tr, fa, t1_w, t1_f)
    fat[fat < 0] = 0
    water[water < 0] = 0
    
    # Clean up temp dirs
    shutil.rmtree(tmp)
    return fat, water



def _predict_mask_numpy(model, op_phase, in_phase, tmp):
    
    input_folder = os.path.join(tmp, 'input_folder')
    predictions = os.path.join(tmp, 'predictions')
    output_folder = os.path.join(tmp, 'output_folder')
    os.makedirs(input_folder)
    os.makedirs(predictions)
    os.makedirs(output_folder)

    # Save numpy arrays as nifti
    case_id = "dixon"
    file_op = os.path.join(input_folder, f"{case_id}_0000.nii.gz")
    file_ip = os.path.join(input_folder, f"{case_id}_0001.nii.gz")
    nifti_op = nib.Nifti1Image(op_phase, np.eye(4))
    nifti_ip = nib.Nifti1Image(in_phase, np.eye(4))
    nib.save(nifti_op, file_op)
    nib.save(nifti_ip, file_ip)

    # Create predictions in a temporary output_folder
    _predict_mask_folder(model, input_folder, output_folder, predictions)
    #__predict_mask_folder(model, input_folder, output_folder)

    # Return result as binary numpy array
    mask_file = os.path.join(output_folder, f"{case_id}.nii.gz")
    waterdom = nib.load(mask_file).get_fdata().astype(np.int8)

    return waterdom


# def __predict_mask_folder(model, input_folder, predictions):

#     # TOD: consider API for numpy arrays to avoid read and write to temp folders

#     # Initialize predictor
#     plans_dir = os.path.join(
#         model, "Dataset001_FatWaterPredictor",
#         "nnUNetTrainer__nnUNetPlans__3d_fullres"
#     )
#     predictor = nnUNetPredictor()
#     predictor.initialize_from_trained_model_folder(plans_dir, None)
#     predictor.predict_from_files(input_folder, predictions)

#     # Skip the postprocessing - not managed to get this to work 
#     # yet in the python API but should be fixable.


def _predict_mask_folder(model, input_folder, output_folder, predictions):

    # These two variables are not used but we are setting to a 
    # dummy value to silence the warnings
    os.environ["nnUNet_raw"] = input_folder 
    os.environ["nnUNet_preprocessed"] = input_folder

    # Folder containing the model weights
    os.environ["nnUNet_results"] = model
    
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

    retcode = process.wait()
    if retcode != 0:
        raise RuntimeError(f"Prediction failed with exit code {retcode}")
    
    # Run post-processing
    # os.makedirs(output_folder, exist_ok=True)
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

    retcode = process.wait()
    if retcode != 0:
        raise RuntimeError(f"Postprocessing failed with exit code {retcode}")



def _compute_fatwater(waterdom, op_phase, in_phase, te_o, te_i, t2s_w, t2s_f, tr, fa, t1_w, t1_f):

    Eof, Eif, Eow, Eiw = 1, 1, 1, 1

    if te_o is not None:
        # Add T2* correction
        Eof = np.exp(-te_o/t2s_f)
        Eif = np.exp(-te_i/t2s_f)
        Eow = np.exp(-te_o/t2s_w)
        Eiw = np.exp(-te_i/t2s_w)

    if tr is not None:
        # Add T1 correction
        ef = np.exp(-tr/t1_f)
        ew = np.exp(-tr/t1_w)
        cfa = np.cos(np.deg2rad(fa))
        Af = (1 - ef) / (1 - cfa * ef)
        Aw = (1 - ew) / (1 - cfa * ew)
        Eof *= Af
        Eif *= Af
        Eow *= Aw
        Eiw *= Aw

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

