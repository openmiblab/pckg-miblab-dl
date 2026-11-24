
from miblab_data.zenodo import fetch as zenodo_fetch
from miblab_dl import get_cache_dir


def fetch(dataset='Dixon'):

    if dataset=='Dixon':
        testdata_dir = get_cache_dir('test_data_pc_dixon')
        testdata = 'test_data_post_contrast_dixon.zip'
        testdatadoi = '15489381'

        # Download ZIP file to cache directory
        folder = zenodo_fetch(testdata, testdata_dir, testdatadoi, extract=True)

        return folder

    raise ValueError(f"{dataset} is not known. Currently only 'Dixon' is available.")