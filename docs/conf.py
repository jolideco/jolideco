# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config


# -- Project information -----------------------------------------------------
# The full version, including alpha/beta/rc tags
from jolideco import __version__

project = "Jolideco"
copyright = "2022, Axel Donath"
author = "Axel Donath"
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinxemoji.sphinxemoji",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/", None)}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/jolideco/jolideco",
    "use_repository_button": True,
}

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "jolideco-logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = ''

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "{0} v{1}".format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + "doc"

# Prefixes that are ignored for sorting the Python module index
modindex_common_prefix = ["jolideco."]


# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", project + ".tex", project + " Documentation", author, "manual")
]


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", project.lower(), project + " Documentation", [author], 1)]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# By default, when rendering docstrings for classes, sphinx.ext.autodoc will
# make docs with the class-level docstring and the class-method docstrings,
# but not the __init__ docstring, which often contains the parameters to
# class constructors across the scientific Python ecosystem. The option below
# will append the __init__ docstring to the class-level docstring when rendering
# the docs. For more options, see:
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"


sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "user/tutorials",  # path to where to save gallery generated output
    "doc_module": ("jolideco",),
    "exclude_implicit_doc": {},
    "filename_pattern": "\.py",
    "reset_modules": ("matplotlib",),
    "download_all_examples": True,
    "capture_repr": ("_repr_html_", "__repr__"),
    "nested_sections": False,
    "min_reported_time": 10,
    "show_memory": False,
    "line_numbers": False,
    "reference_url": {
        # The module you locally document uses None
        "jolideco": None,
    },
}

bibtex_bibfiles = ["references.bib"]


sphinxemoji_style = "twemoji"
