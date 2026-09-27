project = "iful"
author = "William Sheu"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",     
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_copybutton",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}

nb_execution_mode = "off"

html_theme = "furo"
html_logo = "assets/logo.png"
html_static_path = []
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
