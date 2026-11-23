from pathlib import Path
import platformdirs
import shutil

# Define standard user cache directory for dcmri
CACHE_DIR = Path(platformdirs.user_cache_dir(appname="miblab_dl"))


def get_cache_dir(dir=None) -> Path:
    """Ensure cache directory exists and return its Path.

    Args:
        dir (str, optional): Name of a subdirectory of the cache
            directory. If provided, it is created if it does not yet
            exist and its path is returned instead of the cache root.

    Returns:
        Path: the cache directory, or the requested subdirectory.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if dir is None:
        return CACHE_DIR

    subdir = CACHE_DIR / dir
    subdir.mkdir(exist_ok=True)
    return subdir


def clear_cache(dir=None) -> None:
    """Delete the cache directory or one of its subdirectories.

    Args:
        dir (str, optional): Name of a subdirectory of the cache
            directory. If provided, only that subdirectory is deleted.
            If None, the entire cache directory is deleted.
    """
    target = CACHE_DIR if dir is None else CACHE_DIR / dir
    if target.exists():
        shutil.rmtree(target)