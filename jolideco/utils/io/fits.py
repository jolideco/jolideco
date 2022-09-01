from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jolideco.models import FluxComponent, FluxComponents


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

    hdus = [fits.PrimaryHDU()]

    for name, component in result.components.items():
        if component.wcs:
            header = component.wcs.to_header()
        else:
            header = fits.Header()

        header["UPSAMPLE"] = component.upsampling_factor
        hdu = fits.ImageHDU(
            header=header,
            data=component.flux_upsampled_numpy,
            name=f"{name.upper()}",
        )
        hdus.append(hdu)

    for name, component in result.components_init.items():
        if component.wcs:
            header = component.wcs.to_header()
        else:
            header = fits.Header()

        header["UPSAMPLE"] = component.upsampling_factor
        hdu = fits.ImageHDU(
            header=header,
            data=component.flux_upsampled_numpy,
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


IO_FORMATS_READ = {"fits": read_from_fits}
IO_FORMATS_WRITE = {"fits": write_to_fits}
