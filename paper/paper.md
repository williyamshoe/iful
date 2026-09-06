---
title: 'IFUL: A Python package for joint-modeling strong gravitational lensing and their source kinematics'
tags:
  - Python
  - astronomy
  - gravitational lensing
  - spectroscopy
authors:
  - name: William Sheu
    orcid: 0000-0003-1889-0227
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Department of Physics and Astronomy, University of California, Los Angeles, CA 90095, USA
   index: 1
date: 3 September 2026
bibliography: paper.bib
---

# Summary

`IFUL` (Integral Field Unit Lensing) is an open-source Python package for joint modeling and simulating strong gravitational lensing systems together with the spatially resolved internal kinematics of background source galaxy. Strong gravitational lensing occurs when the gravitational field of a massive foreground galaxy or galaxy cluster (the "lens") bends light emitted by a background galaxy (the "source"), magnifying and distorting its appearance into multiple images or arcs. This phenomenon serves as a primary probe in observational astrophysics for measuring cosmological parameters (e.g., the Hubble constant $H_0$, the dark energy equation of state $w$ and the matter density of the universe $\Omega_{\rm m, 0}$), testing dark matter models, and studying high-redshift galaxy structure and evolution across time.

While traditional gravitational lens modeling tools rely primarily on two-dimensional imaging data, `IFUL` incorporates three-dimensional Integral Field Unit (IFU) spectroscopic datacubes into an end-to-end forward-modeling framework. Integral field spectroscopy captures two-dimensional spatial images where every pixel (spaxel) contains a full spectrum, providing spatially resolved measurements of the internal stellar motion of the galaxy. This allows for our framework to not only flux, but also dynamical information into our model. `IFUL` forward-models every spaxel in the 3D datacube by unifying macro lens mass profiles with physical models of source galaxy kinematics. By leveraging spatially resolved kinematic markers within lensed arcs and isolating source line emission from foreground lens light, `IFUL` provides tighter constraints on both lens mass distributions and high-redshift source dynamics than just using the imaging data alone.

# Statement of need

Strong gravitational lens modeling has entered a regime where precision cosmography and mass profile determinations are primarily limited by systematic modeling uncertainties rather than statistical noise. Traditional lens modeling pipelines rely on two-dimensional photometric imaging. While three-dimensional Integral Field Unit (IFU) spectroscopy—from space and ground-based observatories such as *JWST* (NIRSpec, MIRI), VLT (MUSE, ERIS), Keck (KCWI/KCRM), and ALMA—is routinely acquired for lensing fields, these datacubes are typically underutilized. In conventional workflows, IFU data are restricted to measuring integrated deflector stellar velocity dispersions or confirming source galaxy redshifts, discarding the rich three-dimensional spatial-spectral information contained in individual spaxels.

`IFUL` addresses several critical physical and methodological challenges in strong gravitational lensing:
- **Kinematic Markers as Constraints**: When a source galaxy exhibits coherent rotation or dynamic structure, velocity gradients serve as spatial markers across multiple lensed images. Mapping these kinematic features in the image plane provides powerful complementary constraints on the lens mass profile.
- **Continuum Subtraction & Image Disambiguation**: By performing continuum subtraction across the 3D datacube, `IFUL` isolates narrow emission lines belonging to the background source galaxy. This removes severe flux contamination from the bright foreground lens, unblending overlapping components and revealing faint or hidden counter-images that might otherwise remain buried under lens continuum light.
- **Systematic Uncertainties & Degeneracies**: As lens modeling becomes systematic-dominated, `IFUL` provides a novel, independent modeling avenue. Jointly modeling spatial flux and source velocity fields ($v_{\mathrm{los}}$ and $\sigma_v$) helps break classic mass profile degeneracies (such as radial slope and mass-sheet transformations) that cannot be resolved by 2D imaging alone. General improvements to lens mass modeling directly benefit diverse science goals, including time-delay cosmography ($H_0$), dark matter substructure searches, and high-redshift galaxy kinematic structure.

Despite the growing availability of high-resolution IFU observations from current facilities and upcoming Extremely Large Telescope IFU instruments (such as ELT/HARMONI, ELT/MICADO, and TMT/IRIS), there is currently no existing open-source software dedicated to full 3D forward-modeling of lensed IFU datacubes. Individual research groups attempting 3D lens modeling must write custom, non-standardized codebases from scratch, posing a major barrier to entry and hindering scientific reproducibility. `IFUL` fills this software gap by delivering an open-source, modular, and user-friendly Python package built to interface with established lens modeling tools like `lenstronomy` [@lenstronomy2018; @lenstronomy2021].

`IFUL` is designed for astronomers, astrophysicists, and observational cosmologists working on strong gravitational lensing, high-redshift galaxy kinematics, and 3D spectroscopic analysis.

# State of the field



# Software design



# Research impact statement



# AI usage disclosure



# Acknowledgements



# References
