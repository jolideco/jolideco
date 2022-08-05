from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table


def write_to_fits(result, filename, overwrite):
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
    if result.wcs:
        header = result.wcs.to_header()
    else:
        header = None

    primary_hdu = fits.PrimaryHDU(
        header=header,
        data=result.flux,
    )

    init_hdu = fits.ImageHDU(
        header=header,
        data=result.flux_init,
        name="FLUX_INIT",
    )

    table = result.trace_loss.copy()
    table.meta = None
    trace_hdu = fits.BinTableHDU(table, name="TRACE_LOSS")

    config_hdu = fits.BinTableHDU(result.config_table, name="CONFIG")

    hdulist = fits.HDUList(
        [primary_hdu, init_hdu, trace_hdu, config_hdu]
    )

    hdulist.writeto(filename, overwrite=overwrite)


def read_from_fits(filename):
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
    flux = hdulist["PRIMARY"].data
    flux_init = hdulist["FLUX_INIT"].data

    return {
        "config": config,
        "flux_upsampled": flux,
        "flux_init": flux_init,
        "trace_loss": trace_loss,
        "wcs": wcs,
    }


IO_FORMATS_READ = {"fits": read_from_fits}
IO_FORMATS_WRITE = {"fits": write_to_fits}
