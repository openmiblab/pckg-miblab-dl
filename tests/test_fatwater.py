import os
import shutil

import dbdicom as db
import numpy as np

from miblab import zenodo_fetch
import miblab_dl as dl

from plot import volume_to_mosaic


def test_fatwater():
    
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    testdata = 'test_data_post_contrast_dixon.zip'
    testdatadoi = '15489381'

    # Download ZIP file to temp directory
    folder = zenodo_fetch(testdata, tmp_dir, testdatadoi, extract=True)

    # Read DICOM data
    series_op = db.series(folder, contains='out_phase')[0]
    series_ip = db.series(folder, contains='in_phase')[0]
    
    op_phase = db.volume(series_op).values
    in_phase = db.volume(series_ip).values

    fat_recon, wat_recon = dl.fatwater(op_phase, in_phase)

    series_fat = db.series(folder, contains='fat')[0]
    series_water = db.series(folder, contains='water')[0]
    
    fat = db.volume(series_fat).values
    wat = db.volume(series_water).values   

    fat_recon_err = 100 * np.linalg.norm(fat_recon-fat)/np.linalg.norm(fat)
    wat_recon_err = 100 * np.linalg.norm(wat_recon-wat)/np.linalg.norm(wat)

    assert np.round(fat_recon_err) == 9
    assert np.round(wat_recon_err) == 16

    # w_clip = [0, 0.6*min([wat.max(), wat_recon.max()])]
    # volume_to_mosaic(wat, save_as=os.path.join(tmp_dir, f'water.png'), clip=w_clip)
    # volume_to_mosaic(wat_recon, save_as=os.path.join(tmp_dir, f'water_recon.png'), clip=w_clip)
    # f_clip = [0, 0.6*min([fat.max(), fat_recon.max()])]
    # volume_to_mosaic(fat, save_as=os.path.join(tmp_dir, f'fat.png'), clip=f_clip)
    # volume_to_mosaic(fat_recon, save_as=os.path.join(tmp_dir, f'fat_recon.png'), clip=f_clip)



if __name__=='__main__':
    test_fatwater()
    print('fatwater passed all tests..')
