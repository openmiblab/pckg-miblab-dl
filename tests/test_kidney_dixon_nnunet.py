import dbdicom as db
import numpy as np
import vreg

import miblab_dl as dl
from miblab_dl import kidney_dixon


def test_kidney_dixon():
    # Download ZIP file to temp directory
    folder = dl.fetch('Dixon')

    # Read DICOM data in the correct order. 
    desc = ['out_phase', 'in_phase', 'water', 'fat']
    dicom_series = [db.series(folder, contains=d)[0] for d in desc]
    volumes = [db.volume(series, verbose=0) for series in dicom_series]

    # Convert to a 4D input array
    dixon_array = np.stack([v.values for v in volumes], axis=-1)

    kidney_labels = kidney_dixon(dixon_array, volumes[0].spacing, verbose=True)
    assert np.sum(kidney_labels==1) == 70753



if __name__=='__main__':

    test_kidney_dixon()
    #test_kidney_dixon_unetr()

    print('kidney_dixon passed all tests..')
