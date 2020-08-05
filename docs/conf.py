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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import os
import sys
import subprocess
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

# Set the doc generator environment variable
os.environ['DOC_GEN'] = 'True'

curr_dir = os.path.abspath(os.path.dirname(__file__))
python_path = os.path.join(curr_dir, "../python")
sys.path.insert(0, python_path)

project = 'cvm-runtime'
copyright = '2020, CortexLabs Foundation'
author = 'CortexLabs Foundation'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",

    "recommonmark",
    "sphinx_markdown_tables",

    # c++ doxygen
    "breathe",

    "sphinx_rtd_theme",
]

source_parser = {
    ".md": CommonMarkParser,
}
source_suffix = [ ".rst", ".md" ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['Thumbs.db', '.DS_Store']

master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "_static/img/icon.png"

# doxygen -------------------------------------
breathe_projects = {"cvm-runtime": "doxygen_output/xml"}
breathe_default_project = "cvm-runtime"

# hook for doxygen
def run_doxygen():
    """Run the doxygen make command in the designated folder."""
    base_dir = os.path.join(
        os.path.dirname(__file__),
        "..")
    try:
        retcode = subprocess.call(
            "cd %s; doxygen docs/Doxyfile" % base_dir, shell=True)
        if retcode < 0:
            sys.stderr.write(
                "doxygen terminated by signal %s" % (-retcode))
    except OSError as e:
        sys.stderr.write("doxygen execution failed: %s" % e)

github_doc_root = 'https://github.com/CortexFoundation/cvm-runtime'

def setup(app):
    app.add_config_value('recommonmark_config', {
           'url_resolver': lambda url: github_doc_root + url,
           'enable_math': True,
           'enable_inline_math': True,
           'enable_eval_rst': True,
           'enable_auto_toc_tree': True,
           }, True)
    app.add_transform(AutoStructify)

    run_doxygen()

