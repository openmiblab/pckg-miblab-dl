import numpy as np
import dbdicom as db

import miblab_dl as dl


def test_mrseg():
    # Download ZIP file to temp directory
    folder = dl.fetch('Dixon')

    # Read DICOM data in the correct order. 
    dicom_series = db.series(folder, contains='out_phase')[0]
    volume = db.volume(dicom_series, verbose=0)

    label_img = dl.mrseg(volume.values, affine=volume.affine)
    
    assert np.sum(label_img) == 73161506


if __name__=='__main__':
    test_mrseg()
    print('mrseg passed all tests..')
