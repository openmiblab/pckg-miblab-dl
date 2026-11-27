from pathlib import Path
import shutil
import tempfile
from typing import Optional, Sequence, Union
import nibabel as nib
import numpy as np
import vreg
import torch
from mrsegmentator import inference


def mrseg(
    input_array: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    affine: Optional[np.ndarray] = None,
    ref: str = "LPS+",
    verbose=False,
    folds=[0],
) -> np.ndarray:
    """Thin wrapper for MRSegmentator operating on NumPy arrays.

    Source: `MRSegmentator <https://github.com/hhaentze/MRSegmentator>`_.

    Args:
        input_array (np.ndarray): 3D MRI volume to be segmented.
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
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.
        folds (list, optional): Lits of folds to use for the prediction.
            Select any from [0, 1, 2, 3, 4]. Defaults to [0].

    Returns:
        np.ndarray:
            3D array containing integer segmentation labels (0–40) matching the
            shape of ``input_array``. See the `MRSegmentator README
            <https://github.com/hhaentze/MRSegmentator#label-overview>`_
            for the mapping from label values to anatomical structures.

    Example:
        Segment a 3D MRI NumPy array with isotropic voxels:

        >>> import numpy as np
        >>> import miblab_dl as dl
        >>> array = np.zeros((128, 128, 64), dtype=np.float32)
        >>> labels = dl.mrseg(array)
    """
    if affine is None:
        if spacing is None:
            affine = np.eye(4)
        else:
            if len(spacing) != 3:
                raise ValueError(
                    f"spacing must have 3 elements (x, y, z), got {len(spacing)}"
                )
            affine = np.diag([*spacing, 1.0])

    vol = vreg.volume(input_array, affine).convert(ref, "RAS+")
    affine = vol.affine

    input_img = nib.Nifti1Image(input_array, affine)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_file = tmp_path / "input.nii.gz"
        out_file = tmp_path / "input_seg.nii.gz"

        nib.save(input_img, in_file)
        inference.infer(
            images=[str(in_file)],
            outdir=str(tmp_path),
            postfix="seg",
            folds=folds,
            verbose=verbose,
            cpu_only=not torch.cuda.is_available(),
        )
        seg_img = nib.load(out_file)
        label_image = np.asarray(seg_img.dataobj)

    return label_image


def mrseg_nii(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    verbose=False,
    folds=[0],
) -> Optional[np.ndarray]:
    """Thin wrapper for MRSegmentator operating on NIfTI files.

    Source: `MRSegmentator <https://github.com/hhaentze/MRSegmentator>`_.

    Args:
        input_file (str or pathlib.Path): Path to the input NIfTI file
            (``.nii`` or ``.nii.gz``).
        output_file (str or pathlib.Path, optional): Path to save the resulting
            multilabel NIfTI segmentation mask. If ``None``, the segmentation is
            not written to disk and a NumPy array is returned instead. Defaults
            to ``None``.
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.
        folds (list, optional): Lits of folds to use for the prediction.
            Select any from [0, 1, 2, 3, 4]. Defaults to [0].

    Returns:
        np.ndarray or None:
            If ``output_file`` is ``None``, returns a 3D array containing integer
            segmentation labels (0–40) matching the shape of the input volume.
            Otherwise, writes the mask to ``output_file`` and returns ``None``.
            See the `MRSegmentator README
            <https://github.com/hhaentze/MRSegmentator#label-overview>`_
            for the mapping from label values to anatomical structures.

    Example:
        Segment a NIfTI file and return the label mask as a NumPy array:

        >>> import miblab_dl as dl
        >>> labels = dl.mrseg_nii('path/to/file.nii.gz')
    """
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NIfTI file does not exist: {input_path}")

    stem = input_path.name.removesuffix(".gz").removesuffix(".nii")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        generated_file = tmp_path / f"{stem}_seg.nii.gz"

        inference.infer(
            images=[str(input_path)],
            outdir=str(tmp_path),
            postfix="seg",
            verbose=verbose,
            folds=folds,
            cpu_only=not torch.cuda.is_available(),
        )

        if output_file is None:
            seg_img = nib.load(generated_file)
            return np.asarray(seg_img.dataobj)

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(generated_file), str(out_path))

    return None




def mrseg_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    verbose=False,
    folds=[0],
) -> None:
    """Thin batch wrapper for MRSegmentator over a directory of NIfTI files.

    Iterates through all ``.nii`` and ``.nii.gz`` files in ``input_dir`` and
    saves a multilabel segmentation mask (``<stem>_seg.nii.gz``) for each scan
    in ``output_dir``. Existing output files are skipped.

    Source: `MRSegmentator <https://github.com/hhaentze/MRSegmentator>`_.

    Args:
        input_dir (str or pathlib.Path): Path to the directory containing input
            NIfTI volumes (``.nii`` or ``.nii.gz``).
        output_dir (str or pathlib.Path): Path to the directory where output
            multilabel NIfTI masks (``<stem>_seg.nii.gz``) will be saved.
            Created automatically if it does not exist.
        verbose (bool, optional): If True, print logging messages.
            Defaults to False.
        folds (list, optional): Lits of folds to use for the prediction.
            Select any from [0, 1, 2, 3, 4]. Defaults to [0].

    Returns:
        None: Results are written directly to ``output_dir`` as multilabel
        NIfTI files. See the `MRSegmentator README
        <https://github.com/hhaentze/MRSegmentator#label-overview>`_
        for the mapping from label values to anatomical structures.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist or is not a directory.

    Example:
        Segment all MRI NIfTI volumes in a folder and save masks to an output
        folder:

        >>> import miblab_dl as dl
        >>> dl.mrseg_batch(
        ...     input_dir='path/to/nifti_folder',
        ...     output_dir='path/to/output_folder',
        ... )
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect list of files that are not yet computed
    pending_files = []
    for nifti_path in sorted(input_dir.iterdir()):
        if not nifti_path.is_file():
            continue
        if not (nifti_path.name.endswith(".nii") or nifti_path.name.endswith(".nii.gz")):
            continue

        stem = nifti_path.name.removesuffix(".gz").removesuffix(".nii")
        out_file = output_dir / f"{stem}_seg.nii.gz"

        if not out_file.exists():
            pending_files.append(str(nifti_path))

    if not pending_files:
        return

    # Compute only pending files
    inference.infer(
        images=pending_files,
        outdir=str(output_dir),
        postfix="seg",
        verbose=verbose,
        folds=folds,
        cpu_only=not torch.cuda.is_available(),
    )