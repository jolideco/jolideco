import logging
from pathlib import Path
from jolideco.utils.misc import recursive_update

log = logging.getLogger(__name__)


def write_flux_component_to_asdf(flux_component, filename, overwrite, **kwargs):
    """Write flux components to ASDF file

    Parameters
    ----------
    flux_component : `FluxComponent`
        Flux component to serialize to FITS file
    filename : `Path`
        Output filename
    overwrite : bool
        Overwrite file.
    **kwargs : dict
        Keyword arguments passed to `~asdf.AsdfFile.write_to`
    """
    import asdf

    data = flux_component.to_dict(include_data="numpy")

    # Create the ASDF file object from our data tree
    af = asdf.AsdfFile(data)

    path = Path(filename)

    if path.exists() and not overwrite:
        raise OSError(f"{path} already exists!")

    log.info(f"writing {path}")
    # Write the data to a new file
    af.write_to(path, **kwargs)


def read_flux_component_from_asdf(filename):
    """Read flux component from ASDF file

    Parameters
    ----------
    filename : `Path`
        Output filename

    Returns
    -------
    flux_component : `FluxComponent`
        Flux component
    """
    import asdf
    from jolideco.models import SpatialFluxComponent

    path = Path(filename)

    with asdf.open(path, copy_arrays=True) as af:
        data = recursive_update({}, af)
        return SpatialFluxComponent.from_dict(data=data)


def write_flux_components_to_asdf(flux_components, filename, overwrite, **kwargs):
    """Write flux components to ASDF file

    Parameters
    ----------
    flux_components : `FluxComponents`
        Flux components to serialize to FITS file
    filename : `Path`
        Output filename
    overwrite : bool
        Overwrite file.
    **kwargs : dict
        Keyword arguments passed to `~asdf.AsdfFile.write_to`
    """
    write_flux_component_to_asdf(
        flux_component=flux_components, filename=filename, overwrite=overwrite, **kwargs
    )


def read_flux_components_from_asdf(filename):
    """Read flux components from ASDF file

    Parameters
    ----------
    filename : `Path`
        Output filename

    Returns
    -------
    flux_components : `FluxComponents`
        Flux components
    """
    import asdf
    from jolideco.models import FluxComponents

    path = Path(filename)

    with asdf.open(path, copy_arrays=True) as af:
        data = recursive_update({}, af)
        return FluxComponents.from_dict(data=data)


def write_map_result_to_asdf(result, filename, overwrite, **kwargs):
    """Write MAP result to ASDF.

    Parameters
    ----------
    result : `MAPDeconvolverResult`
        Deconvolution result.
    filename : `Path`
        Output filename
    overwrite : bool
        Overwrite file.
    **kwargs : dict
        Keyword arguments passed to `~asdf.AsdfFile.write_to`
    """
    import asdf

    data = {}
    data["components"] = result.components.to_dict(include_data="numpy")
    data["components-init"] = result.components_init.to_dict(include_data="numpy")
    data["trace-loss"] = result.trace_loss
    data["config"] = result.config

    # Create the ASDF file object from our data tree
    af = asdf.AsdfFile(data)

    path = Path(filename)

    if path.exists() and not overwrite:
        raise OSError(f"{path} already exists!")

    log.info(f"writing {path}")
    # Write the data to a new file
    af.write_to(path, **kwargs)


def read_map_result_from_asdf(filename):
    """Read Jolideco result from ASDF.

    Parameters
    ----------
    filename : `Path`
        Output filename

    Returns
    -------
    result : dict
       Dictionary with init parameters for `MAPDeconvolverResult`
    """
    import asdf
    from jolideco.core import MAPDeconvolverResult
    from jolideco.models import FluxComponents

    path = Path(filename)

    log.info(f"Reading {filename}")

    with asdf.open(path, copy_arrays=True) as af:
        data = recursive_update({}, af)
        components = FluxComponents.from_dict(data=data["components"])
        components_init = FluxComponents.from_dict(data=data["components-init"])
        config = data["config"]
        trace_loss = data["trace-loss"]

    return MAPDeconvolverResult(
        config=config,
        components=components,
        components_init=components_init,
        trace_loss=trace_loss,
    )
