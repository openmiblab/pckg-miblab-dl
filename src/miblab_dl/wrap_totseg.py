from pathlib import Path
from typing import Optional, Sequence, Union
import nibabel as nib
import numpy as np
import vreg
from totalsegmentator.python_api import totalsegmentator



def totseg(
    input_array: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    affine: Optional[np.ndarray] = None,
    ref: str = "LPS+",
    task: str = "total",
    verbose=False,
) -> np.ndarray:
    """Thin wrapper for TotalSegmentator operating on NumPy arrays.

    Source: `TotalSegmentator <https://github.com/wasserth/TotalSegmentator>`_.

    Args:
        input_array (np.ndarray): 3D image volume to be segmented.
        spacing (Sequence[float], optional): Voxel spacing ``(sx, sy, sz)`` in
            the x, y, and z directions. Ignored if ``affine`` is provided. If
            neither ``spacing`` nor ``affine`` is provided, isotropic voxels of
            size 1.0 are assumed. Defaults to ``None``.
        affine (np.ndarray, optional): 4x4 affine transformation matrix mapping
            voxel coordinates to world space. Defaults to ``None`` (constructed
            from ``spacing`` or a 4x4 identity matrix).
        ref (str, optional): Anatomical world coordinate system of ``input_array``
            and ``affine`` (e.g., ``'LPS+'``, ``'RAS+'``). Converted to ``'RAS+'``
            for NIfTI compatibility. Defaults to ``'LPS+'``, which is the
            system used by DICOM and ITK.
        task (str, optional): TotalSegmentator segmentation task to run (e.g.,
            ``'total'``, ``'total_mr'``). Defaults to ``'total'``.
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.

    Returns:
        np.ndarray:
            3D array containing integer segmentation labels matching the shape of
            ``input_array``. See the `TotalSegmentator label mapping
            <https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py>`_
            for the dictionary mapping label values to anatomical structures.

    Example:
        Segment a 3D NumPy array with isotropic voxels:

        >>> import numpy as np
        >>> import miblab_dl as dl
        >>> array = np.zeros((128, 128, 64), dtype=np.float32)
        >>> labels = dl.totseg(array, task='total_mr')
    """
    if affine is None:
        if spacing is None:
            affine = np.eye(4)
        else:
            if len(spacing) != 3:
                raise ValueError(f"spacing must have 3 elements (x, y, z), got {len(spacing)}")
            affine = np.diag([*spacing, 1.0])

    vol = vreg.volume(input_array, affine).convert(ref, "RAS+")
    affine = vol.affine

    input_img = nib.Nifti1Image(input_array, affine)
    seg_img = totalsegmentator(input_img, task=task, ml=True, quiet=not verbose)
    label_image = np.asarray(seg_img.dataobj)

    return label_image





def totseg_nii(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    task: str = "total",
    verbose=False,
) -> Optional[np.ndarray]:
    """Thin wrapper for TotalSegmentator operating on NIfTI files.

    Source: `TotalSegmentator <https://github.com/wasserth/TotalSegmentator>`_.

    Args:
        input_file (str or pathlib.Path): Path to the input NIfTI file
            (``.nii`` or ``.nii.gz``).
        output_file (str or pathlib.Path, optional): Path to save the resulting
            multilabel NIfTI segmentation mask. If ``None``, the segmentation is
            not written to disk and a NumPy array is returned instead. Defaults
            to ``None``.
        task (str, optional): TotalSegmentator segmentation task to run (e.g.,
            ``'total'``, ``'total_mr'``). Defaults to ``'total'``.
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.

    Returns:
        np.ndarray or None:
            If ``output_file`` is ``None``, returns a 3D array containing integer
            segmentation labels matching the shape of the input volume.
            Otherwise, writes the mask to ``output_file`` and returns ``None``.
            See the `TotalSegmentator label mapping
            <https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py>`_
            for the dictionary mapping label values to anatomical structures.

    Example:
        Segment a NIfTI file and return the label mask as a NumPy array:

        >>> import miblab_dl as dl
        >>> labels = dl.totseg_nii('path/to/file.nii.gz', task='total_mr')
    """
    if output_file is None:
        seg_img = totalsegmentator(input=input_file, output=None, ml=True, task=task)
        return np.asarray(seg_img.dataobj)

    totalsegmentator(
        input=input_file,
        output=output_file,
        ml=True,
        task=task,
        quiet=not verbose,
    )
    return None




def totseg_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    task: str = "total",
    verbose=False,
) -> None:
    """Thin batch wrapper for TotalSegmentator over a directory of NIfTI files.

    Iterates through all ``.nii`` and ``.nii.gz`` files in ``input_dir`` and
    saves a multilabel segmentation mask (``<stem>_seg.nii.gz``) for each scan
    in ``output_dir``. Existing output files are skipped.

    Source: `TotalSegmentator <https://github.com/wasserth/TotalSegmentator>`_.

    Args:
        input_dir (str or pathlib.Path): Path to the directory containing input
            NIfTI volumes (``.nii`` or ``.nii.gz``).
        output_dir (str or pathlib.Path): Path to the directory where output
            multilabel NIfTI masks (``<stem>_seg.nii.gz``) will be saved.
            Created automatically if it does not exist.
        task (str, optional): TotalSegmentator segmentation task to run (e.g.,
            ``'total'``, ``'total_mr'``). Defaults to ``'total'``.
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.

    Returns:
        None: Results are written directly to ``output_dir`` as multilabel
        NIfTI files. See the `TotalSegmentator label mapping
        <https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py>`_
        for the dictionary mapping label values to anatomical structures.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist or is not a directory.

    Example:
        Segment all MRI NIfTI volumes in a folder and save masks to an output
        folder:

        >>> import miblab_dl as dl
        >>> dl.totseg_batch(
        ...     input_dir='path/to/nifti_folder',
        ...     output_dir='path/to/output_folder',
        ...     task='total_mr',
        ... )
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
    )

    for nifti_path in nifti_files:
        stem = nifti_path.name.removesuffix(".gz").removesuffix(".nii")
        out_file = output_dir / f"{stem}_seg.nii.gz"

        if out_file.exists():
            continue  # Skip already processed scans

        totalsegmentator(
            input=nifti_path,
            output=out_file,
            ml=True,
            task=task,
            quiet=not verbose,
        )