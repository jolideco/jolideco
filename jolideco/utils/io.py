from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


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

    hdus = [fits.PrimaryHDU()]

    for name, flux in result.fluxes_upsampled.items():
        hdu = fits.ImageHDU(
            header=header,
            data=flux,
            name=f"{name.upper()}",
        )
        hdus.append(hdu)

    for name, flux in result.fluxes_init.items():
        hdu = fits.ImageHDU(
            header=header,
            data=flux,
            name=f"{name.upper()}_INIT",
        )

    table = result.trace_loss.copy()
    table.meta = None
    trace_hdu = fits.BinTableHDU(table, name="TRACE_LOSS")
    hdus.append(trace_hdu)

    config_hdu = fits.BinTableHDU(result.config_table, name="CONFIG")
    hdus.append(config_hdu)

    hdulist = fits.HDUList(hdus=hdus)

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

    fluxes, fluxes_init = {}, {}

    for hdu in hdulist:
        if isinstance(hdu, fits.ImageHDU):
            name = hdu.name.replace("_INIT", "").lower()
            if "INIT" in hdu.name:
                fluxes_init[name] = hdu.data
            else:
                fluxes[name] = hdu.data

    return {
        "config": config,
        "fluxes_upsampled": fluxes,
        "fluxes_init": fluxes_init,
        "trace_loss": trace_loss,
        "wcs": wcs,
    }


IO_FORMATS_READ = {"fits": read_from_fits}
IO_FORMATS_WRITE = {"fits": write_to_fits}
