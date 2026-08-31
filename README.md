<p align="center">
  <img src="docs/assets/logo.png" alt="IFUL Logo" width="220" />
</p>

<h1 align="center">IFUL: Integral Field Unit Lensing</h1>

<p align="center">
  <strong>A Python pipeline for joint-modeling strong gravitational lensing and source kinematics</strong>
</p>

<p align="center">
  <a href="https://github.com/williyamshoe/iful/actions/workflows/test.yml"><img src="https://github.com/williyamshoe/iful/actions/workflows/test.yml/badge.svg" alt="Tests Status"></a>
  <a href="https://pypi.org/project/iful/"><img src="https://img.shields.io/pypi/v/iful.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python Versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

## Statement of Need

Strong gravitational lensing is a cornerstone probe of observational cosmology—enabling independent measurements of the Hubble constant ($H_0$) to break the Hubble tension via time-delay cosmography, constraining the dark energy equation-of-state ($w$) and matter density ($\Omega_m$) via compound lensing systems, and probing dark matter substructure and high-redshift galaxy structures. All of these cosmological and astrophysical applications fundamentally depend on the ability to precisely and accurately model the lensing mass distribution.

Traditional lens modeling pipelines rely almost exclusively on 2D imaging data (lensed image positions and brightness arcs), occasionally supplemented by 1D integrated slit spectroscopy of the deflector galaxy. However, imaging-only pipelines face key limitations:
1. **Mass-Sheet and Profile Degeneracies**: Imaging constraints alone often leave the radial mass slope and mass profile parameters underspecified.
2. **Lens/Source Light Contamination**: Bright deflector galaxy light can overlap with source arcs and obscure faint, demagnified central images inside the Einstein radius that strongly constrain the inner mass slope.

**`IFUL` (Integral Field Unit Lensing)** overcomes these limitations by introducing an **end-to-end forward-modeling framework** that incorporates the spatially resolved dynamics of the source galaxy directly into the macro lens model. Observed with modern Integral Field Spectroscopy instruments (e.g., *JWST* NIRSpec, VLT MUSE, Keck KCWI/KCRM, and OSIRIS), `IFUL` forward-models every individual spatial pixel (spaxel) in the 3D IFU datacube from a joint parameterization of:
- **Macro Lens Mass & Light Profiles** (via [`lenstronomy`](https://github.com/lenstronomy/lenstronomy), or binned profiles),
- **Source Line-of-Sight Velocity Fields ($v_{\mathrm{los}}$)** (Arctan, Tanh, Multi-parameter, or binned profiles),
- **Source Velocity Dispersion Fields ($\sigma_v$)** (Exponential, constant, Keplerian, or binned profiles).

By leveraging dynamical markers within lensed arcs as additional kinematic constraints, isolating source emission line light to completely remove lens-light contamination, and reconstructing unlensed source dynamics across multiple lensed images, **`IFUL`** provides tighter constraints on lens mass profiles, profile slopes, and high-redshift source dynamics than imaging data alone.

---

## Key Features

- **Joint Lensing & Kinematics Modeling**: Simultaneously model 2D image-plane lensing light/mass distributions and 3D IFU datacube velocity and dispersion fields.
- **Flexible Kinematic Profiles**: Support for standard kinematic models including Arctan velocity curves, Tanh profiles, multi-parameter profiles, central supermassive black hole (BH) influence, and custom dispersion profiles.
- **Datacube Processing (`ImageSet`)**: Built-in continuum subtraction, outlier rejection (IQR), custom aperture masking, white-light image collapse, and noise estimation.
- **Fast Linear Inversion**: Accelerated linear solver for source plane flux maps to reduce MCMC/optimization overhead.
- **Simulation API (`simulation_api`)**: Easily generate mock JWST NIRSpec or ground-based IFU lensed galaxy datacubes with realistic PSF convolution and instrument noise for mock challenges and forecast studies.
- **Power Binning Support**: Integrated adaptive Power spatial binning (an improvement to traditional Voronoi binning) using `powerbin`.

---

## Installation

### Option 1: Install via `pip` (Recommended)

```bash
pip install iful
```

### Option 2: Install from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/williyamshoe/iful.git
cd iful
pip install -e .
```

To install with testing dependencies:

```bash
pip install -e .[test]
```

---

## Quickstart Example

Here is a quick example creating a mock lensed IFU datacube, configuring the lensing and kinematics models, and evaluating model residuals:

```python
import numpy as np
from iful.simulation_api import SimulationMockImageSet, create_simulation_models

# 1. Create a mock IFU ImageSet (e.g. 10x10 spaxels, 20 wavelength channels)
wavelengths = np.linspace(5000, 5200, 20)
mock_psf_path = "path/to/psf.npy"  # 2D numpy array PSF

imset = SimulationMockImageSet(
    size=10,
    pixscale_arcsec=0.05,
    zs=3.0,
    wavelengths_full=wavelengths,
    psf_path=mock_psf_path,
)

# Set rest-frame emission line peaks (e.g., Lyman-alpha or H-alpha)
imset.restwave_peaks = [1216.0]
imset.init_spec_fit = [3.0, 10.0, 100.0, 1.0]  # [z, sigma, amp, ratio]

# 2. Initialize FlatModel and IFULModel with custom kinematic profiles
profiles = ["ARCTAN", "CONSTANT_FITTED_BH", "SERSIC"]
flat_model, iful_model = create_simulation_models(
    imset,
    theta_E=0.8,
    source_x=0.05,
    source_y=0.05,
    iful_profiles=profiles,
)

# 3. Evaluate model log-likelihood & residuals
num_free_params = iful_model.get_num_free_params()
initial_params = np.zeros(num_free_params)
initial_params[0] = 0.8  # theta_E

chi2_residual = iful_model.generate_residuals(initial_params)
print(f"Chi-squared residual: {chi2_residual:.3f}")
```

For comprehensive tutorials, check out the notebooks in the [`examples/`](examples/) directory.

---

## Repository & Package Structure

```
iful/
├── docs/assets/logo.png     # IFUL package logo
├── examples/                # Jupyter notebook tutorials (excluded from pip package)
│   ├── s4c_init.ipynb
│   ├── s4c_iful_modeling_pl_bh.ipynb
│   └── simulate_lensed_galaxy.ipynb
├── src/iful/                # Package source code (~100 KB lightweight install)
│   ├── __init__.py
│   ├── image_set.py         # 3D Datacube processing & masking
│   ├── flat_modeling.py     # Lensing model wrapper
│   ├── iful_modeling.py     # Joint lensing + IFU kinematics model
│   ├── simulation_api.py    # Mock dataset creation & FITS export
│   └── util.py              # Mathematical profiles & utilities
└── tests/                   # Automated pytest suite
```

---

## Running Tests

To run the automated test suite locally:

```bash
pytest -v --cov=iful
```

---

## Contributing

Contributions, bug reports, and feature requests are welcome! Please feel free to open an issue or submit a pull request on GitHub.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on guidelines and local setup.

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Citation

If you use **IFUL** in your research, please cite:

TODO
