# Example Notebooks

This directory contains demonstration notebooks for simulating mock IFU observations and performing step-by-step 3D datacube lensing and kinematic modeling on real astronomical data (MUSE observations of the Carousel lens, `s4c`).

---

## 1. Mock Data Simulation

* **`simulate_lensed_galaxy.ipynb`**
  - Generates a synthetic 3D lensed galaxy datacube configured for **JWST NIRSpec G235M** observations.
  - Demonstrates lensing galaxy surface brightness evaluation, critical curve/caustic calculation, adding instrument noise, and exporting to FITS format.

---

## 2. Step-by-Step Modeling Workflow (MUSE Carousel Lens Data)

The `s4c_*` notebooks provide an end-to-end pipeline for modeling real MUSE IFU data of the Carousel lens system [Sheu et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad65d3). The recommended running order is as follows:

### **1. `s4c_init.ipynb`**
* **Datacube Initialization & Preprocessing**
* Initializes the 3D datacube and performs spaxel-by-spaxel background continuum subtraction to isolate the lensed source galaxy signal.

### **2. `s4c_flat_modeling_pl.ipynb`**
* **2D Photometric Modeling (Lenstronomy)**
* Performs 2D photometric image modeling on the white-light collapsed image using Lenstronomy.
* Generates an initial parameterization using a primary power-law (EPL) lens mass convergence and a single Sersic source light profile.

### **2.5. `s4c_flat_shapelets_pl.ipynb`** *(Optional)*
* **Extended 2D Source Modeling with Shapelets**
* Builds off `s4c_flat_modeling_pl.ipynb` by incorporating an additional shapelets component to model non-Sersic source light features.

### **3. `s4c_iful_modeling_pl_bh.ipynb`**
* **3D Datacube & Kinematic Modeling (`IFULModel`)**
* Performs 3D IFU datacube modeling using the 2D lens parameterization from `s4c_flat_modeling_pl.ipynb` as a starting point.
* Models the source galaxy's 3D kinematic profiles, including Line-of-Sight (LOS) velocity, velocity dispersion (with central supermassive black hole potential), and flux distribution.

### **4. `s4c_compare_dist.ipynb`**
* **Model Posterior Comparison & Visualization**
* Plots and compares parameter posterior distributions, corner plots, and goodness-of-fit metrics across different model parameterizations.
