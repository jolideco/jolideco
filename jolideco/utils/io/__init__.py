from pathlib import Path

from .asdf import (
    read_flux_component_from_asdf,
    read_flux_components_from_asdf,
    read_map_result_from_asdf,
    write_flux_component_to_asdf,
    write_flux_components_to_asdf,
    write_map_result_to_asdf,
)
from .fits import (
    read_flux_component_from_fits,
    read_flux_components_from_fits,
    read_map_result_from_fits,
    read_npred_calibrations_from_fits,
    write_flux_component_to_fits,
    write_flux_components_to_fits,
    write_map_result_to_fits,
    write_npred_calibrations_to_fits,
)
from .yaml import (
    read_flux_component_from_yaml,
    read_flux_components_from_yaml,
    read_npred_calibrations_from_yaml,
    write_flux_component_to_yaml,
    write_flux_components_to_yaml,
    write_npred_calibrations_to_yaml,
)

__all__ = [
    "guess_format_from_filename",
    "IO_FORMATS_MAP_RESULT_READ",
    "IO_FORMATS_MAP_RESULT_WRITE",
    "IO_FORMATS_FLUX_COMPONENT_READ",
    "IO_FORMATS_FLUX_COMPONENT_WRITE",
    "IO_FORMATS_SPARSE_FLUX_COMPONENT_WRITE",
    "IO_FORMATS_SPARSE_FLUX_COMPONENT_READ",
]


class document_io_formats:
    """Document IO formats decorator

    Requires the {formats} placeholder to be define in the decorated function.

    Attributes
    ----------
    registry : dict
        Registry dictionary
    """

    def __init__(self, registry):
        self.registry = set(registry)

    def __call__(self, func):
        func.__doc__ = func.__doc__.format(formats=self.registry)
        return func


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


def get_writer(filename, format, registry):
    """Dispatch reader based on format and registry

    Parameters
    ----------
    filename : str or `Path`
        Filename
    format : str
        Format definition
    registry : dict
       Registry to use

    Returns
    -------
    writer : func
        Writer function
    """
    filename = Path(filename)

    if format is None:
        format = guess_format_from_filename(filename=filename)

    if format not in registry:
        raise ValueError(
            f"Not a valid format '{format}', choose from " f"{list(registry)}"
        )

    return registry[format]


def get_reader(filename, format, registry):
    """Dispatch reader based on format and registry

    Parameters
    ----------
    filename : str or `Path`
        Filename
    format : str
        Format definition
    registry : dict
       Registry to use

    Returns
    -------
    reader : func
        Reader function
    """
    filename = Path(filename)

    if format is None:
        format = guess_format_from_filename(filename=filename)

    if format not in registry:
        raise ValueError(
            f"Not a valid format '{format}', choose from " f"{list(registry)}"
        )

    return registry[format]


IO_FORMATS_MAP_RESULT_READ = {
    "fits": read_map_result_from_fits,
    "asdf": read_map_result_from_asdf,
}

IO_FORMATS_MAP_RESULT_WRITE = {
    "fits": write_map_result_to_fits,
    "asdf": write_map_result_to_asdf,
}


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

IO_FORMATS_SPARSE_FLUX_COMPONENT_WRITE = {
    "fits": write_flux_component_to_fits,
}

IO_FORMATS_SPARSE_FLUX_COMPONENT_READ = {
    "fits": read_flux_component_from_fits,
}

IO_FORMATS_FLUX_COMPONENTS_READ = {
    "fits": read_flux_components_from_fits,
    "asdf": read_flux_components_from_asdf,
    "yaml": read_flux_components_from_yaml,
}

IO_FORMATS_FLUX_COMPONENTS_WRITE = {
    "fits": write_flux_components_to_fits,
    "asdf": write_flux_components_to_asdf,
    "yaml": write_flux_components_to_yaml,
}

IO_FORMATS_NPRED_CALIBRATIONS_READ = {
    "yaml": read_npred_calibrations_from_yaml,
    "fits": read_npred_calibrations_from_fits,
}

IO_FORMATS_NPRED_CALIBRATIONS_WRITE = {
    "yaml": write_npred_calibrations_to_yaml,
    "fits": write_npred_calibrations_to_fits,
}
