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


def write_flux_component_to_yaml(flux_component, filename, overwrite, filename_data):
    """Write flux component to YAML file

    Parameters
    ----------
    flux_component : `FluxComponent`
        Flux component
    filename : str or `Path`
        Filename

    """
    meta = flux_component.to_dict()

    meta["flux_upsampled"] = filename_data

    path = Path(filename)

    if path.exists() and not overwrite:
        raise IOError(f"{filename} already exists!")

    flux_component.write(filename_data, overwrite=overwrite)
    log.info(f"writing {filename_data}")

    with path.open("w") as f:
        data_str = to_yaml_str(data=meta)
        log.info(f"writing {filename}")
        f.write(data_str)
