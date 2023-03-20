import logging
import numpy as np
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
    "shape": "SHAPE",
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
    "prior.norm.alpha": "PNORMALP",
    "prior.norm.beta": "PNORMBET",
    "prior.patch_norm.type": "PNPTYPE",
    "prior.device": "PDEVICE",
}


FITS_META_INVERSE = {value: key for key, value in FITS_META.items()}


def sparse_flux_component_to_table_hdu(flux_component, name):
    """Convert a sparse flux component to table HDU

    Parameters
    ----------
    flux_component : `SparseFluxComponent`
        Flux component to serialize to a table HDU
    name : str
        Name of the HDU

    Returns
    -------
    hdu : `~astropy.io.fits.BinTableHDU`
        Table HDU
    """
    if flux_component.wcs:
        header = flux_component.wcs.to_header()
    else:
        header = fits.Header()

    data = flux_component.to_dict()

    table = Table()
    table["x_pos"] = np.atleast_1d(data.pop("x_pos"))
    table["y_pos"] = np.atleast_1d(data.pop("y_pos"))
    table["flux"] = np.atleast_1d(data.pop("flux"))

    shape = data.pop("shape")
    header["IMSHAPE1"] = shape[-2]
    header["IMSHAPE2"] = shape[-1]

    meta = flatten_dict(data, sep=META_SEP)

    for key, value in meta.items():
        fits_key = FITS_META[key]
        header[fits_key] = value

    return fits.BinTableHDU(
        data=table,
        header=header,
        name=f"{name.upper()}",
    )


def sparse_flux_component_from_table_hdu(hdu):
    """Create flux component from image HDU

    Parameters
    ----------
    hdu : `~astropy.io.fits.BinTableHDU`
        Image HDU

    Returns
    -------
    flux_component : `SparseFluxComponent`
        Sparse flux component
    """
    from jolideco.models import SparseSpatialFluxComponent

    table = Table.read(hdu)

    shape = (table.meta["IMSHAPE1"], table.meta["IMSHAPE2"])

    return SparseSpatialFluxComponent.from_numpy(
        x_pos=table["x_pos"].data,
        y_pos=table["y_pos"].data,
        flux=table["flux"].data,
        shape=shape,
        use_log_flux=table.meta["LOG_FLUX"],
        frozen=table.meta["FROZEN"],
    )


def flux_component_to_image_hdu(flux_component, name):
    """Convert a flux component into and image HDU

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
    from jolideco.models import SpatialFluxComponent

    data = {}
    data["wcs"] = WCS(hdu.header)
    data["flux_upsampled"] = hdu.data

    for fits_key, key in FITS_META_INVERSE.items():
        value = hdu.header.get(fits_key, None)
        if value is not None:
            data[key] = value

    data = unflatten_dict(data, sep=META_SEP)
    return SpatialFluxComponent.from_dict(data=data)


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
        name = name + name_suffix

        if component.is_sparse:
            hdu = sparse_flux_component_to_table_hdu(
                flux_component=component, name=name
            )
        else:
            hdu = flux_component_to_image_hdu(flux_component=component, name=name)

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
        name = hdu.name.replace(SUFFIX_INIT, "").lower()

        if name in ["config", "trace_loss", "calibrations"]:
            continue

        if isinstance(hdu, fits.ImageHDU):
            component = flux_component_from_image_hdu(hdu=hdu)
        elif isinstance(hdu, fits.BinTableHDU):
            component = sparse_flux_component_from_table_hdu(hdu=hdu)
        else:
            continue

        flux_components[name] = component

    return flux_components


def npred_calibrations_to_table(npred_calibrations):
    """Convert NPredCalibrations to table

    Parameters
    ----------
    npred_calibrations : `NPredCalibrations`
        NPred calibrations

    Returns
    -------
    table : `~astropy.table.Table`
        Table
    """
    data = npred_calibrations.to_dict()

    rows = []

    for name, value in data.items():
        row = {"name": name}
        row.update(value)
        rows.append(row)

    return Table(rows)


def npred_calibrations_from_table(table):
    """Create NPredCalibrations from table

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table


    Returns
    -------
    npred_calibrations : `NPredCalibrations`
        NPred calibrations
    """
    from jolideco.models import NPredCalibrations

    data = {}

    for row in table:
        data_row = dict(zip(row.colnames, row.as_void()))
        name = data_row.pop("name")

        if isinstance(name, bytes):
            name = name.decode("utf-8")

        data[name] = data_row

    return NPredCalibrations.from_dict(data=data)


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
    flux_component : `FluxComponent` or `SparseFluxComponent`
        Flux component to serialize to FITS file
    filename : `Path`
        Output filename
    overwrite : bool
        Overwrite file.
    """
    hdulist = fits.HDUList()

    if flux_component.is_sparse:
        hdu = sparse_flux_component_to_table_hdu(
            flux_component=flux_component, name="primary"
        )
    else:
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
        hdu = hdulist[hdu_name]

        if isinstance(hdu, (fits.ImageHDU, fits.PrimaryHDU)):
            return flux_component_from_image_hdu(hdu=hdu)
        elif isinstance(hdu, fits.BinTableHDU):
            return sparse_flux_component_from_table_hdu(hdu=hdu)


def read_npred_calibrations_from_fits(filename):
    """Read npred calibrations from FITS file

    Parameters
    ----------
    filename : str or `Path`
        Filename

    Returns
    -------
    npred_calibrations : `NPredCalibrations`
        NPred calibrations
    """
    log.info(f"Reading {filename}")

    table = Table.read(filename)
    return npred_calibrations_from_table(table=table)


def write_npred_calibrations_to_fits(npred_calibrations, filename, overwrite):
    """Write npred calibrations to FITS file

    Parameters
    ----------
    npred_calibrations : `NPredCalibrations`
        NPred calibrations
    filename : str or `Path`
        Filename
    overwrite : bool
        Overwrite file.
    """
    table = npred_calibrations_to_table(npred_calibrations)
    table.write(filename, overwrite=overwrite, format="fits")


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

    if result.calibrations:
        table = npred_calibrations_to_table(result.calibrations)
        hdu = fits.BinTableHDU(table, name="CALIBRATIONS")
        hdulist.append(hdu)

        table = npred_calibrations_to_table(result.calibrations_init)
        hdu = fits.BinTableHDU(table, name="CALIBRATIONS" + SUFFIX_INIT)
        hdulist.append(hdu)

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

        hdulist_components = [hdu for hdu in hdulist if SUFFIX_INIT not in hdu.name]
        components = flux_components_from_hdulist(hdulist=hdulist_components)

        hdulist_components = [hdu for hdu in hdulist if SUFFIX_INIT in hdu.name]
        components_init = flux_components_from_hdulist(hdulist=hdulist_components)

        if "CALIBRATIONS" in hdulist:
            table = Table.read(hdulist["CALIBRATIONS"])
            calibrations = npred_calibrations_from_table(table=table)
        else:
            calibrations = None

        if "CALIBRATIONS" + SUFFIX_INIT in hdulist:
            table = Table.read(hdulist["CALIBRATIONS" + SUFFIX_INIT])
            calibrations_init = npred_calibrations_from_table(table=table)
        else:
            calibrations_init = None

    return MAPDeconvolverResult(
        config=config,
        components=components,
        components_init=components_init,
        calibrations=calibrations,
        calibrations_init=calibrations_init,
        trace_loss=trace_loss,
    )
