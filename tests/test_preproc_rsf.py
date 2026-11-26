import dbdicom as db
import numpy as np

import miblab_dl as dl


def test_rsf_candidates():
    # Download ZIP file to temp directory
    folder = dl.fetch('Dixon')

    # Read DICOM data in the correct order. 
    desc = ['out_phase', 'in_phase', 'water', 'fat']
    dicom_series = [db.series(folder, contains=d)[0] for d in desc]
    volumes = [db.volume(series, verbose=0) for series in dicom_series]

    # Convert to a 4D input array
    dixon_array = np.stack([v.values for v in volumes], axis=-1)

    # Compute kidney labels
    kidneys = dl.kidney_dixon(dixon_array, verbose=True)

    # Compute rsf_candidates
    water, fat = dixon_array[..., 2], dixon_array[..., 3]
    rsf_candidates = dl.rsf_candidates(fat, water, kidneys, fat_threshold=0.5)

    print(np.sum(rsf_candidates > 0))
    # assert np.sum(label_image > 0) == 70753



if __name__=='__main__':

    test_rsf_candidates()

    print('rsf_candidates passed all tests..')
