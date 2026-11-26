import dbdicom as db
import numpy as np
import vreg

import miblab_dl as dl
from miblab_dl import kidney_compartments_dixon


def test_kidney_compartments_dixon():
    # Download ZIP file to temp directory
    folder = dl.fetch('Dixon')

    # Read DICOM data in the correct order. 
    desc = ['out_phase', 'in_phase', 'water', 'fat']
    dicom_series = [db.series(folder, contains=d)[0] for d in desc]
    volumes = [db.volume(series, verbose=0) for series in dicom_series]

    # Convert to a 4D input array and read spacing
    array = np.stack([v.values for v in volumes], axis=-1)

    label_image = kidney_compartments_dixon(array, verbose=True)
    print(np.sum(label_image > 0))



if __name__=='__main__':

    test_kidney_compartments_dixon()

    print('kidney_compartments_dixon passed all tests..')
