# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join('..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print('sys.path', sys.path)
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LORE_sa'
copyright = '2023, Kode srl'
author = 'Kode srl'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.coverage",
              'sphinx.ext.viewcode',
              "sphinx.ext.napoleon",'sphinx.ext.duration',
              'sphinx.ext.autosummary'
              ]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'
# The master toctree document.
master_doc = 'index'
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

autodoc_default_options = {"members": True, "inherited-members": True}
autosummary_generate = True
autoclass_content='class'

# Mock imports for dependencies that are not needed for documentation building
autodoc_mock_imports = [
    'numpy', 'pandas', 'scipy', 'sklearn', 'joblib', 'imblearn',
    'deap', 'bitarray', 'category_encoders', 'skmultilearn', 'arff'
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "style.css" will overwrite the builtin "style.css".
html_static_path = ['_static']

html_theme_options = {
    'display_version': True,
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4
}
