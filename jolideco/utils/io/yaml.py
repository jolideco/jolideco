import logging
from pathlib import Path

from ruamel.yaml import YAML

log = logging.getLogger(__name__)

__all__ = ["to_yaml_str"]


def to_yaml_str(data):
    """Convert dict to YAML string

    Parameters
    ----------
    data : dict
        Data dictionary

    Returns
    -------
    yaml_str : str
        YAML string
    """
    yaml = YAML(typ=["rt", "string"])
    yaml.default_flow_style = False
    return yaml.dump_to_string(data)


def write_yaml(filename, data, overwrite):
    """Write dict to YAML file

    Parameters
    ----------
    filename : str or Path
        Filename
    data : dict
        Data to write
    overwrite : bool
        Overwrite file?

    """
    path = Path(filename)

    if path.exists() and not overwrite:
        raise OSError(f"{filename} already exists!")

    with path.open("w") as f:
        data_str = to_yaml_str(data=data)
        log.info(f"Writing {filename}")
        f.write(data_str)


def load_yaml(filename):
    """Load data from YAML file

    Parameters
    ----------
    filename : str or Path
        Filename

    Returns
    -------
    data : dict
        Data from YAML file
    """
    path = Path(filename)

    yaml = YAML()
    yaml.allow_duplicate_keys = True

    with path.open("r") as f:
        log.info(f"Reading {path}")
        data = yaml.load(f)

    return data


def flux_component_to_yaml_dict(flux_component, filename, name=None):
    """Convert flux component to YAML dict

    Parameters
    ----------
    flux_component : `FluxComponent`
        Flux component
    name : str
        Name of the flux component

    Returns
    -------
    data : dict
        YAML dictionary
    """
    data = flux_component.to_dict()
    path = Path(filename)

    if name is None:
        name = path.stem

    filename_data = path.parent / f"{name}-data.fits"

    data["flux_upsampled"] = str(filename_data.absolute())
    return data


def write_flux_component_to_yaml(flux_component, filename, overwrite):
    """Write flux component to YAML file

    Parameters
    ----------
    flux_component : `FluxComponent`
        Flux component
    filename : str or `Path`
        Filename
    overwrite : bool
        Overwrite file.
    """
    data = flux_component_to_yaml_dict(
        flux_component=flux_component,
        filename=filename,
    )

    flux_component.write(data["flux_upsampled"], overwrite=overwrite)
    write_yaml(filename=filename, data=data, overwrite=overwrite)


def write_flux_components_to_yaml(flux_components, filename, overwrite):
    """Write flux components to YAML file

    Parameters
    ----------
    flux_components : `FluxComponents`
        Flux components
    filename : str or `Path`
        Filename
    overwrite : bool
        Overwrite file.
    """
    data = {}

    for name, flux_component in flux_components.items():
        data[name] = flux_component_to_yaml_dict(
            flux_component=flux_component,
            filename=filename,
            name=name,
        )
        filename_data = data[name]["flux_upsampled"]
        flux_component.write(filename_data, overwrite=overwrite)

    write_yaml(filename=filename, data=data, overwrite=overwrite)


def read_flux_component_from_yaml(filename):
    """Read flux component from YAML file

    Parameters
    ----------
    filename : str or `Path`
        Filename

    Returns
    -------
    flux_component : `FluxComponent`
        Flux component
    """
    from jolideco.models import FluxComponent

    data = load_yaml(filename=filename)

    return FluxComponent.from_dict(data=data)


def read_flux_components_from_yaml(filename):
    """Read flux components from YAML file

    Parameters
    ----------
    filename : str or `Path`
        Filename

    Returns
    -------
    flux_components : `FluxComponents`
        Flux components
    """
    from jolideco.models import FluxComponents

    data = load_yaml(filename=filename)

    return FluxComponents.from_dict(data=data)
