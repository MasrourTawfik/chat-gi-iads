# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'CHAT GI-IADS'
copyright = '2024, ENSAM MEKNES'
author = 'ENSAM MEKNES'

release = '0.1'
version = '0.0.1'

# -- General configuration
import sys,os

sys.path.insert(0, os.path.abspath('..'))


extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

html_static_path = ['_static']

