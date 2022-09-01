from ruamel.yaml import YAML

__all__ = ["to_yaml_str"]


def to_yaml_str(data):
    """_summary_

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
