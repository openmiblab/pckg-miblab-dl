from pathlib import Path
import shutil

import numpy as np
import nibabel as nib

from miblab_dl.kidney_dixon_nnunet import kidney_dixon, kidney_dixon_batch
from miblab_dl.rsfseg import rsf_dixon, rsf_dixon_batch


def kidney_compartments_dixon(input_array, spacing=(1.0, 1.0, 1.0), device=None,
                    verbose=False):
    """Segment individual kidney compartments (parenchyma and rsf) on Dixon images.

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
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.

    Returns:
        numpy.ndarray: Label image of the same shape as the first three
        axes of `input_array` (left kidney=1, right kidney=2, left rsf=3, right rsf=4).

    Raises:
        ValueError: If `input_array` or `spacing` has the wrong shape.
    """
    parenchyma = kidney_dixon(input_array, spacing, device, verbose=verbose)
    rsf = rsf_dixon(input_array, spacing, device, verbose=verbose)

    rsf[rsf > 0] += 2 # Increment to make rsf labels 3 and 4

    # combine the two labels, with rsf taking precedent at intersections
    kidney_compartments = np.where(rsf > 0, rsf, parenchyma)

    return kidney_compartments




def _combine_labels(dir_a, dir_b, dir_out=None, dtype=np.uint8):
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    dir_out = Path(dir_out) if dir_out else dir_a.with_name(dir_a.name + "_combined")
    dir_out.mkdir(parents=True, exist_ok=True)

    files = sorted(list(dir_a.glob("*.nii.gz")) + list(dir_a.glob("*.nii")))
    for fa in files:
        fb = dir_b / fa.name
        if not fb.exists():
            print(f"Skipping {fa.name}: no matching file in {dir_b}")
            continue

        imgA, imgB = nib.load(fa), nib.load(fb)
        A = np.asanyarray(imgA.dataobj)
        B = np.asanyarray(imgB.dataobj)

        B[B > 0] += 2 # Increment RSF labels

        if A.shape != B.shape:
            print(f"Skipping {fa.name}: shape mismatch {A.shape} vs {B.shape}")
            continue
        if not np.allclose(imgA.affine, imgB.affine):
            print(f"Warning {fa.name}: affines differ, using the affine of A")

        combined = np.where(B > 0, B, A).astype(dtype)

        hdr = imgA.header.copy()
        hdr.set_data_dtype(dtype)
        nib.save(nib.Nifti1Image(combined, imgA.affine, hdr), dir_out / fa.name)

    # Report files in B that had no partner in A
    names_a = {f.name for f in files}
    for fb in sorted(list(dir_b.glob("*.nii.gz")) + list(dir_b.glob("*.nii"))):
        if fb.name not in names_a:
            print(f"Note: {fb.name} exists in B but not in A")

    return dir_out


    

def kidney_compartments_dixon_batch(input_folder, output_folder, device=None,
                          verbose=False,
                          num_processes_preprocessing=3, num_processes_export=3):
    """Segment individual kidney compartments (parenchyma and rsf) on Dixon images, for many cases on disk.

    The input folder holds one set of four NIfTI files per case, named
    ``<case>_0000.nii.gz`` (out-phase), ``<case>_0001.nii.gz`` (in-phase),
    ``<case>_0002.nii.gz`` (water) and ``<case>_0003.nii.gz`` (fat). A label
    image ``<case>.nii.gz`` is written to the output folder for each case
    (left kidney=1, right kidney=2, left rsf=3, right rsf=4).

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
    output_folder = Path(output_folder)
    parenchyma_output_folder = output_folder.with_name(output_folder.name + "_parenchyma")
    rsf_output_folder = output_folder.with_name(output_folder.name + "_rsf")

    kidney_dixon_batch(input_folder, str(parenchyma_output_folder), device,
                        verbose=verbose,
                        num_processes_preprocessing=num_processes_preprocessing, 
                        num_processes_export=num_processes_export)

    rsf_dixon_batch(input_folder, str(rsf_output_folder), device,
                    verbose=verbose,
                    num_processes_preprocessing=num_processes_preprocessing, 
                    num_processes_export=num_processes_export)

    _combine_labels(str(parenchyma_output_folder), str(rsf_output_folder), output_folder)

    # Remove temporary
    shutil.rmtree(parenchyma_output_folder)
    shutil.rmtree(rsf_output_folder)