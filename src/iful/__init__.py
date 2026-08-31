"""
Integral Field Unit Lensing (IFUL) package.

This package provides tools for modeling and simulating lensed Integral Field Unit (IFU)
datacubes, including 2D photometric modeling, 3D kinematic modeling, and mock simulation APIs.
"""

__version__ = "0.1.0"

from .image_set import ImageSet
from .flat_modeling import FlatModel
from .iful_modeling import IFULModel
from .simulation_api import (
    SimulationMockImageSet,
    create_simulation_models,
    run_galaxy_simulation,
    add_instrument_noise,
    export_to_fits,
)

__all__ = [
    "__version__",
    "ImageSet",
    "FlatModel",
    "IFULModel",
    "SimulationMockImageSet",
    "create_simulation_models",
    "run_galaxy_simulation",
    "add_instrument_noise",
    "export_to_fits",
]

