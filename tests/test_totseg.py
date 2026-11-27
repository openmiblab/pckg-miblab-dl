import numpy as np
import dbdicom as db

import miblab_dl as dl
from miblab_dl import totseg



def test_totseg():
    # Download ZIP file to temp directory
    folder = dl.fetch('Dixon')

    # Read DICOM data in the correct order. 
    dicom_series = db.series(folder, contains='out_phase')[0]
    volume = db.volume(dicom_series, verbose=0)

    label_img = totseg(volume.values, affine=volume.affine, task='total_mr')
    
    assert np.sum(label_img) == 79556255


if __name__=='__main__':
    test_totseg()
    print('totseg passed all tests..')
