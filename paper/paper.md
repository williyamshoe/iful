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

`IFUL` makes several improvements to traditional 2D lens modeling:

- Kinematic markers as constraints: When a source galaxy exhibits coherent rotation or dynamic structure, velocity gradients serve as spatial markers across multiple lensed images. Mapping these kinematic features in the image plane provides powerful complementary constraints on the lens mass profile.

- Continuum subtraction and image disambiguation: By performing continuum subtraction across the 3D datacube, `IFUL` isolates narrow emission lines belonging to the background source galaxy. This removes severe flux contamination from the bright foreground lens, unblending overlapping components and revealing faint or hidden counter-images that might otherwise remain buried under lens continuum light.

- Systematic uncertainties and degeneracies: As lens modeling becomes systematic-dominated, `IFUL` provides a novel, independent modeling avenue. Jointly modeling spatial flux and source velocity fields ($v_{\mathrm{los}}$ and $\sigma_v$) helps break classic mass profile degeneracies (such as radial slope and mass-sheet transformations) that cannot be resolved by 2D imaging alone.

- Understanding source galaxy dynamics: By forward-modeling the macro lens concurrently with the 3D IFU datacube, `IFUL` constructs intrinsic kinematic maps of the background galaxy that integrate spectroscopic information across all multiple lensed images. Combining spaxel data from every image into a unified source-plane model enables a robust determination of the source galaxy's intrinsic line-of-sight velocity ($v_{\mathrm{los}}$) and velocity dispersion ($\sigma_v$) fields. This provides a detailed, high-resolution view of high-redshift galaxy kinematics and dynamics while fully marginalizing over lens-model parameters.

General improvements to lens mass modeling directly benefit diverse science goals, including time-delay cosmography ($H_0$), dark matter substructure searches, and multi-plane dark energy measurements.

Despite the growing availability of high-resolution IFU observations from current facilities and upcoming Extremely Large Telescope IFU instruments (such as ELT/HARMONI, ELT/MICADO, and TMT/IRIS), there is currently no existing open-source software dedicated to full 3D forward-modeling of lensed IFU datacubes. Individual research groups attempting 3D lens modeling must write custom, non-standardized codebases from scratch, posing a major barrier to entry and hindering scientific reproducibility. `IFUL` fills this software gap by delivering an open-source, modular, and user-friendly Python package built to interface with established lens modeling tools like `lenstronomy` [@lenstronomy2018; @lenstronomy2021].

`IFUL` is designed for astronomers, astrophysicists, and observational cosmologists working on strong gravitational lensing, high-redshift galaxy kinematics, and 3D spectroscopic analysis.

# State of the field

Strong gravitational lens mass modeling is supported by a mature ecosystem of open-source software packages in astrophysics. The most widely adopted Python package is `lenstronomy` [@lenstronomy2018; @lenstronomy2021], which provides a flexible, multi-purpose framework for modeling two-dimensional (2D) imaging data, time-delay cosmography, and dark matter substructure. Other notable 2D lens modeling software packages include `PyAutoLens` [@Nightingale2021], `glafic` [@Oguri2010glafic], `gigalens` [@Gu2022], `gravlens`/`lensmodel` [@keeton2001computational], and `GLEE` [@Suyu2010; @Suyu2012]. While these frameworks excel at modeling high-resolution, multi-band photometric images (e.g., from the *Hubble Space Telescope* or *JWST* imaging filters), they operate exclusively on 2D spatial pixel arrays. Consequently, existing tools lack the architectural data structures and forward-modeling engines necessary to process three-dimensional (3D) Integral Field Unit (IFU) spectroscopic datacubes ($x, y, \lambda$), which combine spatial morphology with wavelength-resolved spectral line dynamics.

The scientific advantage of combining 3D IFU spectroscopy with strong gravitational lensing has been recognized in observational cosmology, yet no general-purpose, open-source software package has been released to serve this need:

- Early 3D implementations: @Bolton2007 developed an early 3D IFU lens modeling code in IDL to analyze source velocity fields in strong lens systems. However, this code was proprietary, never publicly released, and was not expanded into a general-purpose software package. Moreover, IFU instrumentation, spatial resolution, and sensitivity have advanced dramatically over the past two decades with modern spectrographs (e.g., VLT/MUSE, Keck/KCWI, *JWST*/NIRSpec).

- Proprietary Bayesian 3D methodologies: @Rizzo2018 introduced a 3D modeling technique designed to recover lensed source kinematics. However, their underlying codebase remains private. Additionally, their study was primarily a methodological demonstration using simulated data; no application of their pipeline to real data was published.

- Two-stage image-plane kinematic mapping: Alternative workflows attempt to bypass full 3D datacube forward-modeling by first extracting 2D kinematic maps (such as line-of-sight velocity $v_{\mathrm{los}}$ and dispersion $\sigma_v$) in the image plane and subsequently ray-tracing those 2D maps back to the source plane [@Chirivi2020; @Zhou2025Kinematic]. While useful, this decoupled approach does not perform end-to-end forward-modeling. Extracting 2D kinematic markers directly from distorted or overlapping lensed arcs introduces significant systematic errors due to spatial blending, beam smearing, continuum contamination, and the destruction of spaxel-level covariances. Furthermore, even this two-stage methodology lacks a publicly available, standardized software package.

A fundamental design question during the development of `IFUL` was whether to contribute 3D modeling features directly upstream to `lenstronomy` or to create `IFUL` as an independent, standalone package. `lenstronomy`'s architecture is tightly coupled to 2D image arrays via its `ImageModel` and `Data` class abstractions. Modifying `lenstronomy`'s core codebase to natively support 3D spectroscopic datacubes—which requires spaxel-by-spaxel spectral line synthesis, 3D velocity dispersion convolution kernels, continuum subtraction, source-plane kinematic parameterizations, and customized 3D tensor likelihood evaluations—would require a structural overhaul of `lenstronomy`'s data structures and API contract.

Instead, `IFUL` adopts an ecosystem wrapper design: it builds directly on top of `lenstronomy`, using it "under-the-hood" as its underlying engine for 2D spatial ray-tracing, coordinate transformations, and lens mass profile calculations (e.g., EPL, NFW, external shear). `IFUL` extends this foundation by providing higher-level orchestrations for 3D datacubes, spectral line synthesis, kinematic field mapping, and joint 3D likelihood computation. This design choice delivers a specialized, modular, and user-friendly 3D modeling tool to the astronomical community while avoiding code duplication and maintaining full interoperability with `lenstronomy`'s mature 2D lensing engine.

# Software design



# Research impact statement



# AI usage disclosure



# Acknowledgements



# References
