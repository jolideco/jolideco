import logging

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jolideco.utils.misc import flatten_dict, unflatten_dict

log = logging.getLogger(__name__)

SUFFIX_INIT = "-INIT"
META_SEP = "."

# TODO: this is incomplete, extend as needed...
FITS_META = {
    "use_log_flux": "LOG_FLUX",
    "upsampling_factor": "UPSAMPLE",
    "frozen": "FROZEN",
    "prior.type": "PTYPE",
    "prior.stride": "PSTRIDE",
    "prior.cycle_spin": "PSPIN",
    "prior.cycle_spin_subpix": "PSUBSPIN",
    "prior.jitter": "PJITTER",
    "prior.alpha": "PALPHA",
    "prior.beta": "PBETA",
    "prior.width": "PWIDTH",
    "prior.gmm.type": "PGMMTYPE",
    "prior.gmm.stride": "PGMMSTRI",
    "prior.norm.type": "PNORMTYP",
    "prior.norm.max_value": "PNORMMAX",
}


FITS_META_INVERSE = {value: key for key, value in FITS_META.items()}


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

    data = flatten_dict(flux_component.to_dict(), sep=META_SEP)

    for key, value in data.items():
        fits_key = FITS_META[key]
        header[fits_key] = value

    return fits.ImageHDU(
        header=header,
        data=flux_component.flux_upsampled_numpy,
        name=f"{name.upper()}",
    )


def flux_component_from_image_hdu(hdu):
    """Create flux component from image HDU

    Parameters
    ----------
    hdu : `~astropy.io.fits.ImageHDU`
        Image HDU

    Returns
    -------
    flux_component : `FluxComponent`
        Flux component to serialize to an image HDU
    """
    from jolideco.models import FluxComponent

    data = {}
    data["wcs"] = WCS(hdu.header)
    data["flux_upsampled"] = hdu.data

    for fits_key, key in FITS_META_INVERSE.items():
        value = hdu.header.get(fits_key)
        if value:
            data[key] = value

    data = unflatten_dict(data, sep=META_SEP)
    return FluxComponent.from_dict(data=data)


def flux_components_to_hdulist(flux_components, name_suffix=""):
    """Convert flux components to hdu list

    Parameters
    ----------
    flux_components : `FluxComponents`
        Flux components
    name_suffix : str
        Suffix to be used for the name

    Returns
    -------
    hdulist : list of `~astropy.io.fits.ImageHDU`
        HDU list
    """
    hdulist = []

    for name, component in flux_components.items():
        hdu = flux_component_to_image_hdu(
            name=name + name_suffix, flux_component=component
        )
        hdulist.append(hdu)

    return hdulist


def flux_components_from_hdulist(hdulist):
    """Create flux components from HDU list

    Parameters
    ----------
    hdulist : `~astropy.io.fits.HDUList`
        HDU list

    Returns
    -------
    flux_components : `FluxComponents`
        Flux components
    """
    from jolideco.models import FluxComponents

    flux_components = FluxComponents()

    for hdu in hdulist:
        if isinstance(hdu, fits.ImageHDU):
            name = hdu.name.replace(SUFFIX_INIT, "").lower()
            component = flux_component_from_image_hdu(hdu=hdu)
            flux_components[name] = component

    return flux_components


def write_flux_components_to_fits(flux_components, filename, overwrite):
    """Write flux components to FITS file

    Parameters
    ----------
    flux_components : `FluxComponents`
        Flux components to serialize to FITS file
    filename : `Path`
        Output filename
    overwrite : bool
        Overwrite file.
    """
    hdulist = fits.HDUList([fits.PrimaryHDU()])
    hdus = flux_components_to_hdulist(flux_components=flux_components)
    hdulist.extend(hdus)
    log.info(f"writing {filename}")
    hdulist.writeto(filename, overwrite=overwrite)


def read_flux_components_from_fits(filename):
    """Write flux components to FITS file

    Parameters
    ----------
    filename : `Path`
        Output filename

    Returns
    -------
    flux_components : `FluxComponents`
        Flux components
    """
    with fits.open(filename) as hdulist:
        return flux_components_from_hdulist(hdulist=hdulist)


def write_flux_component_to_fits(flux_component, filename, overwrite):
    """Write flux component to FITS file

    Parameters
    ----------
    flux_component : `FluxComponent`
        Flux component to serialize to FITS file
    filename : `Path`
        Output filename
    overwrite : bool
        Overwrite file.
    """
    hdulist = fits.HDUList()

    hdu = flux_component_to_image_hdu(flux_component=flux_component, name="primary")
    hdulist.append(hdu)

    log.info(f"writing {filename}")
    hdulist.writeto(filename, overwrite=overwrite)


def read_flux_component_from_fits(filename, hdu_name=0):
    """Write flux component to FITS file

    Parameters
    ----------
    filename : `Path`
        Output filename
    hdu_name : int or str
        HDU name

    Returns
    -------
    flux_component : `FluxComponent`
        Flux component
    """
    with fits.open(filename) as hdulist:
        return flux_component_from_image_hdu(hdu=hdulist[hdu_name])


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

    hdus = flux_components_to_hdulist(result.components, name_suffix=SUFFIX_INIT)
    hdulist.extend(hdus)

    table = result.trace_loss.copy()
    table.meta = None
    trace_hdu = fits.BinTableHDU(table, name="TRACE_LOSS")
    hdulist.append(trace_hdu)

    config_hdu = fits.BinTableHDU(result.config_table, name="CONFIG")
    hdulist.append(config_hdu)

    log.info(f"writing {filename}")
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
    from jolideco.core import MAPDeconvolverResult

    log.info(f"Reading {filename}")

    with fits.open(filename) as hdulist:
        config_table = Table.read(hdulist["CONFIG"])
        config = dict(config_table[0])

        trace_loss = Table.read(hdulist["TRACE_LOSS"])

        hdulist = [hdu for hdu in hdulist if SUFFIX_INIT not in hdu.name]
        components = flux_components_from_hdulist(hdulist=hdulist)

        hdulist = [hdu for hdu in hdulist if SUFFIX_INIT in hdu.name]
        components_init = flux_components_from_hdulist(hdulist=hdulist)

    return MAPDeconvolverResult(
        config=config,
        components=components,
        components_init=components_init,
        trace_loss=trace_loss,
    )
