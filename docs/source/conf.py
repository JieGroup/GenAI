# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme
from datetime import date
import errno
import sphinx.util.osutil
sphinx.util.osutil.ENOENT = errno.ENOENT
source_encoding = 'utf-8-sig'

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GenAI'
# copyright = '2024, Jie Ding'
copyright = f'{date.today().year}, Jie Ding'
author = 'Jie Ding'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',  # if using Markdown
    'sphinx.ext.mathjax',
    'sphinx.ext.imgmath',  # Use this if rendering math as images
    'sphinxcontrib.mermaid',
]

# Set font size for image-based math rendering
imgmath_image_format = 'svg'
imgmath_font_size = 13  # Adjust font size here (in pt)

templates_path = ['_templates']
exclude_patterns = []

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_js_files = [
    'https://code.jquery.com/jquery-3.6.0.min.js',  # Load jQuery
    'https://cdnjs.cloudflare.com/ajax/libs/mermaid/8.13.5/mermaid.min.js',  # Add Mermaid JS
    'custom.js',
]

html_show_sourcelink = False # Hide the "View page source" link
html_show_sphinx = False # Remove footnote

html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'G-ZDQXZS0531',  #  Provided by Google in your dashboard
    'logo_only': False,
    'display_version': False,
    'prev_next_buttons_location': 'both', #'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
