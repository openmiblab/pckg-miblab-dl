import dbdicom as db
import numpy as np
import vreg

from miblab_dl import kidney_pc_dixon, kidney_pc_dixon_unetr, get_cache_dir
from miblab_data.zenodo import fetch as zenodo_fetch


def test_kidney_pc_dixon():

    testdata_dir = get_cache_dir('test_data_pc_dixon')
    testdata = 'test_data_post_contrast_dixon.zip'
    testdatadoi = '15489381'

    # Download ZIP file to temp directory
    folder = zenodo_fetch(testdata, testdata_dir, testdatadoi, extract=True)

    # Read DICOM data
    study = db.studies(folder)[0]
    series = [
        'Dixon_post_contrast_out_phase', 
        'Dixon_post_contrast_in_phase', 
        'Dixon_post_contrast_water',
        'Dixon_post_contrast_fat',
    ]
    arrays = [db.pixel_data(study + [s]) for s in series]
    array = np.stack(arrays, axis=-1)

    mask = kidney_pc_dixon(array, verbose=True)
    assert np.sum(mask==1) == 79285


def test_kidney_pc_dixon_unetr():
    
    testdata_dir = get_cache_dir('test_data_pc_dixon')
    testdata = 'test_data_post_contrast_dixon.zip'
    testdatadoi = '15489381'

    # Download ZIP file to temp directory
    folder = zenodo_fetch(testdata, testdata_dir, testdatadoi, extract=True)

    # Read DICOM data
    study = db.studies(folder)[0]
    series = [
        'Dixon_post_contrast_out_phase', 
        'Dixon_post_contrast_in_phase', 
        'Dixon_post_contrast_water',
        'Dixon_post_contrast_fat',
    ]
    arrays = [db.pixel_data(study + [s]) for s in series]
    array = np.stack(arrays, axis=-1)
    vol = db.volume(study + [series[0]])

    mask = kidney_pc_dixon_unetr(vreg.volume(array, vol.affine), verbose=True)
    assert np.sum(mask.values==1) == 73871


if __name__=='__main__':

    test_kidney_pc_dixon()
    #test_kidney_pc_dixon_unetr()

    print('kidney_pc_dixon passed all tests..')
