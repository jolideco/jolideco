__all__ = ["to_str", "format_class_str"]

TABSIZE = 2
MAX_WIDTH = 24


def to_str(data, level=1):
    """Convert dict to string"""
    if isinstance(data, dict):
        info = "\n\n"
        for key, value in data.items():
            value = to_str(data=value, level=level + 1)
            indent = level * "\t"
            width = MAX_WIDTH - TABSIZE * level
            info += indent + f"{key:{width}s}: {value}\n"
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