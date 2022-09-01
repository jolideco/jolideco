from msilib.schema import Component
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jolideco.models import FluxComponent, FluxComponents


def flux_component_to_image_hdu(flux_component, name):
    """Convert a flux component into and image HDU

    The meta information is just dumbed as a YAML string into the
    FITS header. 

    Parameters
    ----------
    flux_component : `FluxComponent`
        Flux component to serialize to an image HDU
    name : str
        Name of the HDU

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        Image HDU
    """
    if flux_component.wcs:
        header = flux_component.wcs.to_header()
    else:
        header = fits.Header()

    return fits.ImageHDU(
        header=header,
        data=flux_component.flux_upsampled_numpy,
        name=f"{name.upper()}",
    )


def flux_components_to_hdulist(flux_components, name_suffix=""):
    """_summary_

    Parameters
    ----------
    flux_components : `FluxComponents`
        Flux components
    name_suffix : str
        Suffix to be used for the name

    Returns
    -------
    hdulist : `~astropy.io.fits.HDUList`
        HDU list
    """
    hdulist = fits.HDUList()

    for name, component in flux_components.items():
        hdu = flux_component_to_image_hdu(name=name + name_suffix, flux_component=component)
        hdulist.append(hdu)

    return hdulist


def write_map_result_to_fits(result, filename, overwrite):
    """Write MAP result to FITS.

    Parameters
    ----------
    result : `MAPDeconvolverResult`
        Deconvolution result.
    filename : `Path`
        Output filename
    overwrite : bool
        Overwrite file.
    """
    hdulist = fits.HDUList([fits.PrimaryHDU()])

    hdus = flux_components_to_hdulist(result.components)
    hdulist.extend(hdus)

    hdus = flux_components_to_hdulist(result.components, name_suffix="_init")
    hdulist.extend(hdus)

    table = result.trace_loss.copy()
    table.meta = None
    trace_hdu = fits.BinTableHDU(table, name="TRACE_LOSS")
    hdulist.append(trace_hdu)

    config_hdu = fits.BinTableHDU(result.config_table, name="CONFIG")
    hdulist.append(config_hdu)

    hdulist.writeto(filename, overwrite=overwrite)


def read_map_result_from_fits(filename):
    """Read Jolideco result from FITS.

    Parameters
    ----------
    filename : `Path`
        Output filename

    Returns
    -------
    result : dict
       Dictionary with init parameters for `MAPDeconvolverResult`
    """
    hdulist = fits.open(filename)

    wcs = WCS(hdulist["PRIMARY"].header)

    config_table = Table.read(hdulist["CONFIG"])
    config = dict(config_table[0])

    trace_loss = Table.read(hdulist["TRACE_LOSS"])

    components = FluxComponents()
    components_init = FluxComponents()

    for hdu in hdulist:
        if isinstance(hdu, fits.ImageHDU):
            name = hdu.name.replace("_INIT", "").lower()
            if "INIT" in hdu.name:
                components_init[name] = FluxComponent.from_flux_init_numpy(
                    flux_init=hdu.data,
                    upsampling_factor=hdu.header.get("UPSAMPLE", 1),
                )
            else:
                components[name] = FluxComponent.from_flux_init_numpy(
                    flux_init=hdu.data,
                    upsampling_factor=hdu.header.get("UPSAMPLE", 1),
                )

    return {
        "config": config,
        "components": components,
        "components_init": components_init,
        "trace_loss": trace_loss,
        "wcs": wcs,
    }

