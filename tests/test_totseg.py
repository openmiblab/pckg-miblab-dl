import numpy as np
import dbdicom as db

import miblab_dl as dl
from miblab_dl import totseg



def test_totseg():
    folder = dl.fetch('Dixon')

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
