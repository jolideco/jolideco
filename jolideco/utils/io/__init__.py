from pathlib import Path

from .fits import read_map_result_from_fits, write_map_result_to_fits

__all__ = [
    "guess_format_from_filename",
    "IO_FORMATS_MAP_RESULT_READ",
    "IO_FORMATS_MAP_RESULT_WRITE",
    "IO_FORMATS_FLUX_COMPONENT_READ",
    "IO_FORMATS_FLUX_COMPONENT_WRITE",
]


def guess_format_from_filename(filename):
    """Guess I/O format from filename

    Parameters
    ----------
    filename : str or `Path`
        Filename

    Returns
    -------
    format : {"fits", "yaml"}
        Guessed format
    """
    path = Path(filename)

    if path.suffix == ".fits":
        return "fits"
    elif path.suffix in [".yml", ".yaml"]:
        return "yaml"
    else:
        raise ValueError(f"Cannot guess format from filename {filename}")


IO_FORMATS_MAP_RESULT_READ = {"fits": read_map_result_from_fits}
IO_FORMATS_MAP_RESULT_WRITE = {"fits": write_map_result_to_fits}


IO_FORMATS_FLUX_COMPONENT_READ = {}
IO_FORMATS_FLUX_COMPONENT_WRITE = {}
