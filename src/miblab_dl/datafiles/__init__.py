import os
from pathlib import Path

# Setting the environment variables for nnUNET using totalsegmentator paths
# Hack to fix issue that nnUNET gets the environment variables upon first import
# Totalsegmentator sets these variables, but nnunet is imported before totalsegmentator is run.
def get_totalseg_dir():
    if "TOTALSEG_HOME_DIR" in os.environ:
        totalseg_dir = Path(os.environ["TOTALSEG_HOME_DIR"])
    else:
        # in docker container finding home not properly working therefore map to /tmp
        home_path = Path("/tmp") if str(Path.home()) == "/" else Path.home()
        totalseg_dir = home_path / ".totalsegmentator"
    return totalseg_dir

def setup_nnunet():
    # check if environment variable totalsegmentator_config is set
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        weights_dir = os.environ["TOTALSEG_WEIGHTS_PATH"]
    else:
        # in docker container finding home not properly working therefore map to /tmp
        config_dir = get_totalseg_dir()
        # (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(exist_ok=True, parents=True)
        # (config_dir / "nnunet/results/nnUNet/2d").mkdir(exist_ok=True, parents=True)
        weights_dir = config_dir / "nnunet/results"

    # This variables will only be active during the python script execution. Therefore
    # we do not have to unset them in the end.
    os.environ["nnUNet_raw"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_results"] = str(weights_dir)

setup_nnunet()



from miblab_dl.dlseg import totseg
from miblab_dl.dlsegkidney import kidney_pc_dixon, kidney_pc_dixon_unetr
from miblab_dl.fatwatermap import fatwater, clear_cache







