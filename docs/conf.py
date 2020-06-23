# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from github_link import make_linkcode_resolve

sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'epftoolbox'
copyright = '2020, Jesus Lago'
author = 'Jesus Lago'

# The full version, including alpha/beta/rc tags
release = '1.0'

# Specify the master doc, otherwise the build at read the docs fails
master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # Support automatic documentation
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage', # Automatically check if functions are documented
    'sphinx.ext.mathjax',  # Allow support for algebra
    # 'sphinx.ext.viewcode', # Include the source code in documentation
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',             # Support NumPy style docstrings
    "sphinx_rtd_theme",
]

autodoc_default_flags = ['members']
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Function to find the code on the web
# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve('epftoolbox',
                                         'https://github.com/jeslago/epftoolbox/'
                                         'blob/master/'
                                         '{package}/{path}#L{lineno}')