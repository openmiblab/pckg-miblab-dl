import dbdicom as db
import numpy as np
import vreg

import miblab_dl as dl
from miblab_dl import kidney_dixon, kidney_dixon_unetr


def test_kidney_dixon():
    # Download ZIP file to temp directory
    folder = dl.fetch('Dixon')

    # Read DICOM data in the correct order. 
    desc = ['out_phase', 'in_phase', 'water', 'fat']
    dicom_series = [db.series(folder, contains=d)[0] for d in desc]
    volumes = [db.volume(series, verbose=0) for series in dicom_series]

    # Convert to a 4D input array and read spacing
    array = np.stack([v.values for v in volumes], axis=-1)

    mask = kidney_dixon(array, verbose=True)
    assert np.sum(mask==1) == 79285


def test_kidney_dixon_unetr():
    # Download ZIP file to temp directory
    folder = dl.fetch('Dixon')

    # Read DICOM data in the correct order. 
    desc = ['out_phase', 'in_phase', 'water', 'fat']
    dicom_series = [db.series(folder, contains=d)[0] for d in desc]
    volumes = [db.volume(series, verbose=0) for series in dicom_series]

    # Convert to a 4D input array and create 4D volume
    array = np.stack([v.values for v in volumes], axis=-1)
    dixon_vol = vreg.volume(array, volumes[0].affine)

    mask = kidney_dixon_unetr(dixon_vol, verbose=True)
    assert np.sum(mask.values==1) == 73871


if __name__=='__main__':

    test_kidney_dixon()
    #test_kidney_dixon_unetr()

    print('kidney_dixon passed all tests..')
