from collections.abc import Mapping

__all__ = ["to_str", "format_class_str"]

TABSIZE = 2
MAX_WIDTH = 24


def flatten_dict(d, parent_key="", sep="."):
    """Flatten dictionary

    Taken from:

    https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep="."):
    """Unflatten dictionary"

    Taken from https://stackoverflow.com/a/6037657/19802442
    """
    result = dict()
    for key, value in d.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result


def recursive_update(d, u):
    """Recursively update a dict object"""
    for key in reversed(u.keys()):
        if key in ["asdf_library", "history"]:
            continue

        value = u[key]

        if isinstance(value, Mapping):
            d[key] = recursive_update(d.get(key, {}), value)
        else:
            d[key] = value

    return d


def to_str(data, level=1):
    """Convert dict to string"""
    if isinstance(data, dict):
        info = "\n\n"
        for key, value in data.items():
            value = to_str(data=value, level=level + 1)
            indent = level * "\t"
            width = MAX_WIDTH - TABSIZE * level
            info += indent + f"{key:{width}s}: {value}\n"
    elif isinstance(data, float):
        info = f"{data:.3f}"
    else:
        info = str(data)

    return info


def format_class_str(instance):
    """Format class str"""
    cls_name = instance.__class__.__name__
    info = cls_name + "\n"
    info += len(cls_name) * "-"
    data = instance.to_dict()
    info += to_str(data=data, level=1)
    return info.expandtabs(tabsize=TABSIZE)
