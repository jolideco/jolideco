import logging
import warnings

import click

from jolideco import __version__


# We implement the --version following the example from here:
# http://click.pocoo.org/5/options/#callbacks-and-eager-options
def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    print(f"Jolideco version {__version__}")
    ctx.exit()


# http://click.pocoo.org/5/documentation/#help-parameter-customization
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# https://click.palletsprojects.com/en/5.x/python3/#unicode-literals
click.disable_unicode_literals_warning = True


@click.group("jolideco", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--log-level",
    default="info",
    help="Logging verbosity level.",
    type=click.Choice(["debug", "info", "warning", "error"]),
)
@click.option("--ignore-warnings", is_flag=True, help="Ignore warnings?")
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Print version and exit.",
)
def cli(log_level, ignore_warnings):  # noqa: D301
    """Jolideco command line interface (CLI).

    Jolideco is a Python package for joint likelihood deconvolution of low counts data.
    Use ``--help`` to see available sub-commands, as well as the available
    arguments and options for each sub-command.
    """
    logging.basicConfig(level=log_level.upper())

    if ignore_warnings:
        warnings.simplefilter("ignore")


@cli.command("test")
def test():
    from jolideco import test

    test()


if __name__ == "__main__":
    cli()
