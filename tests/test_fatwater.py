import dbdicom as db
import numpy as np

import miblab_dl as dl


def test_fatwater():
    # Download ZIP file to temp directory
    folder = dl.fetch('Dixon')

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
