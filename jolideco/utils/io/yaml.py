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


def write_flux_component_to_yaml(
    flux_component, filename, overwrite, filename_data=None
):
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
    meta = flux_component.to_dict()
    path = Path(filename)

    if filename_data is None:
        filename_data = path.parent / f"{path.stem}-data.fits"

    meta["flux_upsampled"] = str(Path(filename_data).relative_to(path.parent))

    if path.exists() and not overwrite:
        raise OSError(f"{filename} already exists!")

    flux_component.write(filename_data, overwrite=overwrite)

    with path.open("w") as f:
        data_str = to_yaml_str(data=meta)
        log.info(f"writing {filename}")
        f.write(data_str)


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

    path = Path(filename)

    yaml = YAML()
    yaml.allow_duplicate_keys = True

    with path.open("r") as f:
        data = yaml.load(f)

    filename = path.parent / data["flux_upsampled"]
    data["flux_upsampled"] = FluxComponent.read(filename).flux_upsampled

    return FluxComponent.from_dict(data=data)
