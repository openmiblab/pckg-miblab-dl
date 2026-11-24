import dbdicom as db
import numpy as np
import vreg

import miblab_dl as dl
from miblab_dl import rsf_dixon, rsf_dixon_batch


def test_rsf_dixon():
    # Download ZIP file to temp directory
    folder = dl.fetch('Dixon')

    # Read DICOM data in the correct order. 
    desc = ['out_phase', 'in_phase', 'water', 'fat']
    dicom_series = [db.series(folder, contains=d)[0] for d in desc]
    volumes = [db.volume(series, verbose=0) for series in dicom_series]

    # Convert to a 4D input array and read spacing
    array = np.stack([v.values for v in volumes], axis=-1)

    label_image = rsf_dixon(array, verbose=True)

    assert np.sum(label_image==1) == 6515


def test_rsf_dixon_batch(case='Dixon'):
    # Download ZIP file to temp directory
    folder = dl.fetch(case)

    # Read DICOM data in the correct order. 
    desc = ['out_phase', 'in_phase', 'water', 'fat']
    dicom_series = [db.series(folder, contains=d)[0] for d in desc]
    volumes = [db.volume(series, verbose=0) for series in dicom_series]

    batch_folder = dl.get_cache_dir('Dixon_batch')
    for i, volume in enumerate(volumes):
        nifti_file = batch_folder / f'{case}_{i:04d}.nii.gz'
        # for now: save with the incorrect affine
        # When new model is implemented, check that this works correctly
        v = vreg.volume(volume.values, np.eye(4))
        vreg.write_nifti(v, nifti_file)

    output_folder = dl.get_cache_dir('Dixon_batch_results')
    rsf_dixon_batch(batch_folder, output_folder)

    label_image = vreg.read_nifti(output_folder / f'{case}.nii.gz').values
    
    assert np.sum(label_image==1) == 6515



if __name__=='__main__':

    test_rsf_dixon()
    test_rsf_dixon_batch()
    
    print('rsf_dixon passed all tests..')
