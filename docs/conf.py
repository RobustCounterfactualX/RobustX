# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../'))


project = 'robustx'
copyright = '2025, Junqi Jiang, Luca Marzari, Aaryan Purohit, Francesco Leofante'
author = 'Junqi Jiang, Luca Marzari, Aaryan Purohit, Francesco Leofante'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',            # Automatically include docstrings
    'sphinx.ext.napoleon',           # Support for Google/NumPy-style docstrings
    'sphinx_autodoc_typehints',      # Show type hints in docs_old
    'sphinx.ext.viewcode',           # Link to source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
