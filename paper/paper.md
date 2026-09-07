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
    affiliation: 1
affiliations:
 - name: Department of Physics and Astronomy, University of California, Los Angeles, CA 90095, USA
   index: 1
date: 6 September 2026
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

- Understanding source galaxy dynamics: By forward-modeling the macro lens concurrently with the 3D IFU datacube, `IFUL` constructs intrinsic kinematic maps of the background galaxy that integrate spectroscopic information across all multiple lensed images. Combining spaxel data from every image into a unified source-plane model enables a robust determination of the source galaxy's intrinsic line-of-sight velocity and velocity dispersion fields. This provides a detailed, high-resolution view of high-redshift galaxy kinematics and dynamics while fully marginalizing over lens-model parameters.

General improvements to lens mass modeling directly benefit diverse science goals, including time-delay cosmography, dark matter substructure searches, and multi-plane dark energy measurements.

Despite the growing availability of high-resolution IFU observations from current facilities and upcoming Extremely Large Telescope IFU instruments (such as ELT/HARMONI, ELT/MICADO, and TMT/IRIS), there is currently no existing open-source software dedicated to full 3D forward-modeling of lensed IFU datacubes. Individual research groups attempting 3D lens modeling must write custom, non-standardized codebases from scratch, posing a major barrier to entry and hindering scientific reproducibility. `IFUL` fills this software gap by delivering an open-source, modular, and user-friendly Python package built to interface with established lens modeling tools like `lenstronomy` [@lenstronomy2018; @lenstronomy2021].

`IFUL` is designed for astronomers, astrophysicists, and observational cosmologists working on strong gravitational lensing, high-redshift galaxy kinematics, and 3D spectroscopic analysis.

# State of the field

Strong gravitational lens mass modeling is supported by a mature ecosystem of open-source software packages in astrophysics. One of the most widely adopted Python package is `lenstronomy` [@lenstronomy2018; @lenstronomy2021], which provides a flexible, multi-purpose framework for modeling two-dimensional (2D) imaging data, time-delay cosmography, and dark matter substructure. Other notable 2D lens modeling software packages include `PyAutoLens` [@Nightingale2021], `glafic` [@Oguri2010glafic], `gigalens` [@Gu2022], `gravlens`/`lensmodel` [@keeton2001computational], and `GLEE` [@Suyu2010; @Suyu2012]. While these frameworks excel at modeling high-resolution, multi-band photometric images (e.g., from the *Hubble Space Telescope* or *JWST* imaging filters), they operate exclusively on 2D spatial pixel arrays. Consequently, existing tools lack the architectural data structures and forward-modeling engines necessary to process three-dimensional (3D) Integral Field Unit (IFU) spectroscopic datacubes ($x, y, \lambda$), which combine spatial morphology with wavelength-resolved spectral line dynamics.

The potential scientific advantage of combining source kinematic reconstruction with strong gravitational lensing has been acknowledged, yet no general-purpose, open-source software package has been released to serve this need:

- Early 3D implementations: @Bolton2007 developed an early 3D IFU lens modeling code in IDL to analyze source velocity fields in strong lens systems. However, this code was proprietary, never publicly released, and was not expanded into a general-purpose software package. Moreover, IFU instrumentation, spatial resolution, and sensitivity have advanced dramatically over the past two decades with modern spectrographs (e.g., VLT/MUSE, Keck/KCWI, *JWST*/NIRSpec).

- Proprietary Bayesian 3D methodologies: @Rizzo2018 introduced a 3D modeling technique designed to recover lensed source kinematics. Their pipeline share similar properties as our implementation in `IFUL`. However, their underlying codebase remains private, their study was primarily a methodological demonstration using simulated data; no application of their pipeline to real data has been published.

- Two-stage image-plane kinematic mapping: Alternative workflows attempt to bypass full 3D datacube forward-modeling by first extracting 2D kinematic maps (such as $v_{\mathrm{los}}$ and $\sigma_v$) in the image plane and subsequently ray-tracing those 2D maps back to the source plane [@Chirivi2020; @Zhou2025Kinematic]. While useful, this decoupled approach does not perform end-to-end forward-modeling. Extracting 2D kinematic markers directly from distorted or overlapping lensed arcs introduces significant systematic errors due to spatial blending, beam smearing, continuum contamination, and the destruction of spaxel-level covariances. Furthermore, even this two-stage methodology lacks a publicly available, standardized software package.

Currently, `IFUL` offers the only open-source solution to lens modeling 3D IFU datacubes. While our implementation is build with `lenstronomy` as its underlying engine (which we discuss in the section below), we choose to build rather than contribute as its application to 3D IFU datacubes drastically differs from the traditional 2D images.  The differences in input data, necessity of source kinematic models, and a distinct reconstruction framework (to name only a few reasons) are why we do not simply contribute to an existing 2D lens modeling software. 

# Software design

A core architectural principle of `IFUL` is modularity through software reuse. Rather than re-implementing 2D gravitational ray-tracing models—a task for which mature, community-vetted open-source packages already exist—`IFUL` uses `lenstronomy` [@lenstronomy2018; @lenstronomy2021] as its underlying 2D spatial solver engine. This choice grants `IFUL` immediate access to `lenstronomy`'s comprehensive catalog of mass profiles (such as EPL, NFW, SIS, and external shear) while focusing its own scope on extending 2D spatial transformations into three-dimensional ($x, y, \lambda$) spectroscopic datacube space. `IFUL` serves as the coordination layer that organizes these 2D spatial image planes into a unified 3D physical forward-modeling framework.

To complement this modular framework, `IFUL` enforces a strict separation of concerns across its pipeline. Datacube ingestion, spaxel continuum background subtraction, spatial masking, and noise estimation are encapsulated within the `ImageSet` module, completely isolated from lensing physics. Spatial coordinate transformations and 2D ray-tracing abstractions are managed by `FlatModel`, while 3D spatial-spectral synthesis and kinematic model evaluations are orchestrated by `IFULModel`. Decoupling data preprocessing, 2D ray-tracing, physical kinematic modeling, and simulation tools allows researchers to easily integrate custom velocity profiles, alternative continuum subtraction routines, or adaptive spatial binning methods without altering the underlying likelihood or lens-solver engines.

# Research impact statement

`IFUL` has already demonstrated its scientific utility through direct application to observational datasets. In an upcoming study (Sheu et al. in prep.), `IFUL` was deployed to perform joint 3D spatial-spectral lens modeling of the "Carousel" strong gravitational lens system using high-resolution VLT/MUSE IFU observations [@carousel0; @carousel1; @carousel2]. By isolating emission line kinematics from foreground lens continuum light, `IFUL` successfully reconstructed the intrinsic line-of-sight velocity field and velocity dispersion f the background spiral galaxy at redshift $z = 1.432$. Crucially, incorporating spaxel-level spectroscopic constraints yielded tighter posterior probability distributions on macro lens mass model parameters compared to traditional 2D photometric modeling alone, while enabling robust constraints on high-redshift disk dynamics and the central supermassive black hole mass.

Beyond individual lens system modeling, `IFUL` has been presented at international astrophysics conferences, receiving strong interest from the gravitational lensing and galaxy evolution communities. The software is currently being adopted by multiple research groups for ongoing projects, including the 3D spectroscopic modeling of lensed quasar host galaxies and lensed star-forming systems observed with *JWST*/NIRSpec.

# AI usage disclosure

Generative AI (Gemini 3.6 Flash) was used to check formatting, spelling, and grammar of the manuscript, as well as the generation of the docstrings of our code. However, we emphasize that no generative AI was used in the creation of the code itself.

# Acknowledgements

I thank Karl Glazebrook for the initial concept of 3D lens modeling, and thank Xiaosheng Huang, Felipe Urcelay, Sean Xu, Linus Upson, Evan Odell, Tesla Jeltema, Jackson O'Donnell, Aleksandar Cikota, and Tommaso Treu for their continued support in the construction of `IFUL`.

# References
