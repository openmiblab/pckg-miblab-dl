import os
import shutil

import numpy as np
import dbdicom as db

from miblab_dl import totseg, get_cache_dir
from miblab_data.zenodo import fetch as zenodo_fetch



def test_totseg():
    
    testdata_dir = get_cache_dir('test_data_pc_dixon')
    testdata = 'test_data_post_contrast_dixon.zip'
    testdatadoi = '15489381'

    # Download ZIP file to temp directory
    folder = zenodo_fetch(testdata, testdata_dir, testdatadoi, extract=True)

    # Read DICOM data
    # To segment on all volumes:
    # vols = [db.volume(s) for s in db.series(folder)]
    # To segment on a single volume:
    study = db.studies(folder)[0]
    vols = db.volume(study + ['Dixon_post_contrast_water'])

    # Segment using TotalSegmentator
    mask = totseg(
        vols, 
        task='total_mr', 
        roi_subset=['kidney_left', 'kidney_right'], 
        device='cpu',
        fastest=True,
    )
    assert float(np.sum(mask.values==1)) == 71003


if __name__=='__main__':
    test_totseg()
    print('totseg passed all tests..')
