from pathlib import Path

from .asdf import read_flux_component_from_asdf, write_flux_component_to_asdf
from .fits import (
    read_flux_component_from_fits,
    read_map_result_from_fits,
    write_flux_component_to_fits,
    write_map_result_to_fits,
)
from .yaml import read_flux_component_from_yaml, write_flux_component_to_yaml

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
    format : {"fits", "yaml", "asdf"}
        Guessed format
    """
    path = Path(filename)

    if path.suffix == ".fits":
        return "fits"
    elif path.suffix == ".asdf":
        return "asdf"
    elif path.suffix in [".yml", ".yaml"]:
        return "yaml"
    else:
        raise ValueError(f"Cannot guess format from filename {filename}")


IO_FORMATS_MAP_RESULT_READ = {"fits": read_map_result_from_fits}
IO_FORMATS_MAP_RESULT_WRITE = {"fits": write_map_result_to_fits}


IO_FORMATS_FLUX_COMPONENT_READ = {
    "fits": read_flux_component_from_fits,
    "yaml": read_flux_component_from_yaml,
    "asdf": read_flux_component_from_asdf,
}

IO_FORMATS_FLUX_COMPONENT_WRITE = {
    "yaml": write_flux_component_to_yaml,
    "fits": write_flux_component_to_fits,
    "asdf": write_flux_component_to_asdf,
}
