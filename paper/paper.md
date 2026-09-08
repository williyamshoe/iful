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
date: 8 September 2026
bibliography: paper.bib
---

# Summary

`IFUL` (Integral Field Unit Lensing) is an open-source Python package for joint modeling and simulating strong gravitational lensing systems alongside the spatially resolved internal kinematics of background source galaxies. Strong gravitational lensing magnifies and distorts light from a background galaxy (the "source") around a massive foreground object (the "lens") into multiple images and/or arcs. This phenomenon serves as a primary probe in observational astrophysics for measuring cosmological parameters (e.g., the Hubble constant $H_0$, the dark energy equation of state $w$, and the matter density $\Omega_{\rm m, 0}$), testing dark matter models, studying high-redshift galaxy evolution across cosmic time, and many more applications.

While traditional lens modeling tools rely primarily on 2D photometric imaging ($x$ and $y$ axes), `IFUL` incorporates 3D Integral Field Unit (IFU) spectroscopic datacubes ($x$, $y$, and $\lambda$ axes) into an end-to-end forward-modeling framework. Integral field spectroscopy provides a full spectrum for every spatial pixel (spaxel), delivering spatially resolved measurements of stellar and gas kinematics. This allows our framework to incorporate not only spatial flux but also dynamical information into the lens model. `IFUL` forward-models every spaxel in the 3D datacube by unifying macro lens mass profiles with physical models of source galaxy kinematics. By leveraging spatially resolved kinematic markers within lensed arcs and isolating source line emission from foreground lens light, `IFUL` yields competitive constraints on both lens mass distributions than high-resolution photometric imaging data alone, while also providing insight into high-redshift source dynamics.

# Statement of need

Strong gravitational lens modeling is currently in a regime where precision cosmography and mass profile determinations are dominated by systematic modeling uncertainties rather than statistical noise. Traditional lens modeling pipelines rely on 2D photometric imaging. While 3D IFU spectroscopy—from space and ground-based observatories such as *JWST* (NIRSpec, MIRI), VLT (MUSE, ERIS), Keck (KCWI/KCRM), and ALMA—is routinely acquired for lensing fields, these datacubes are typically underutilized. In conventional workflows, IFU data are restricted to measuring integrated deflector stellar velocity dispersions or confirming source galaxy redshifts, while overlooking the rich spectral and spatial information contained in individual spaxels of the source galaxy.

`IFUL` advances lens modeling beyond traditional 2D imaging through several key capabilities:

- Kinematic markers as constraints: When a source galaxy exhibits coherent rotation and/or dynamic structure, velocity gradients (both line-of-sign velocities $v_{\mathrm{los}}$ and velocity dispersions $\sigma_v$) serve as spatial markers across multiple lensed images, providing powerful complementary constraints on the lens mass profile.

- Continuum subtraction and image disambiguation: By performing continuum subtraction across the 3D datacube, `IFUL` isolates narrow emission lines from the background source galaxy. This removes severe flux contamination from the bright foreground lens, unblending overlapping components and revealing faint counter-images otherwise buried under lens continuum light.

- Intrinsic source kinematic reconstruction: Concurrent forward-modeling of the macro lens and 3D IFU datacube synthesizes spaxel data from all lensed images into a unified source-plane model. This enables robust determination of the source galaxy's intrinsic line-of-sight velocity and velocity dispersion fields while fully marginalizing over lens-model parameters.

- Systematic uncertainties: Being an alternative to traditional 2D modeling, `IFUL` offers an independent lens model sourced from datacubes. This serves as an additional probe for systematic uncertainties in a regime of overly-constrained lens models.

Despite the growing availability of high-resolution IFU observations from current facilities and upcoming IFU instruments (such as ELT/HARMONI, ELT/MICADO, and TMT/IRIS), no existing open-source software was dedicated to full 3D forward-modeling of lensed IFU datacubes. Individual research groups were forced to write custom, non-standardized codebases from scratch, creating barriers to entry and hindering scientific reproducibility. `IFUL` fills this software gap by delivering an open-source, modular Python package built to interface with established lens modeling tools like `lenstronomy` [@lenstronomy2018; @lenstronomy2021].

`IFUL` is designed for astronomers, astrophysicists, and observational cosmologists working on strong gravitational lensing, high-redshift galaxy kinematics, and 3D spectroscopic analysis.

# State of the field

Strong gravitational lens mass modeling is supported by a mature ecosystem of open-source software packages. Widely adopted 2D Python packages include `lenstronomy` [@lenstronomy2018; @lenstronomy2021], `PyAutoLens` [@Nightingale2021], `glafic` [@Oguri2010glafic], `gigalens` [@Gu2022], `gravlens` [@keeton2001computational], and `GLEE` [@Suyu2010; @Suyu2012]. While these frameworks excel at modeling high-resolution photometric images (e.g., from *HST* or *JWST*), they operate exclusively on 2D spatial pixel arrays. As such, existing tools lack the data structures and forward-modeling engines necessary to process 3D IFU spectroscopic datacubes.

Although the scientific benefit of combining source kinematic reconstruction with strong gravitational lensing is recognized, existing 3D approaches remain limited:

- Early 3D implementations: @Bolton2007 developed an early 3D IFU lens modeling code in IDL, but it was proprietary, unreleased, and predated modern high-resolution spectrographs (e.g., VLT/MUSE, Keck/KCWI, *JWST*/NIRSpec).

- Proprietary Bayesian 3D methodologies: @Rizzo2018 introduced a 3D modeling technique designed to recover lensed source kinematics. However, their underlying codebase remains private, and their method was demonstrated only on simulated data with no published application to observational data.

- Decoupled two-stage kinematic mapping: Workflows such as @Chirivi2020 and @Zhou2025Kinematic extract 2D kinematic maps ($v_{\mathrm{los}}$, $\sigma_v$) in the image plane before ray-tracing them back to the source plane. While useful, this decoupled approach avoids full 3D forward-modeling and suffers from systematic errors caused by spatial blending, beam smearing, continuum contamination, and lost spaxel-level covariances. Furthermore, no standardized open-source software package exists for this two-stage methodology.

`IFUL` provides the only open-source solution dedicated to 3D IFU datacube lens modeling. While built with `lenstronomy` as its underlying 2D spatial solver, `IFUL` was developed as a dedicated package because 3D datacube forward-modeling requires distinct data structures, source kinematic profiles, and a 3D reconstruction framework.

# Software design

A core architectural principle of `IFUL` is modularity. Rather than re-implementing 2D gravitational ray-tracing models, `IFUL` uses `lenstronomy` as its underlying 2D spatial solver engine. This choice grants `IFUL` immediate access to `lenstronomy`'s comprehensive catalog of mass profiles (e.g., EPL, NFW, SIS, external shear) while focusing its own scope on organizing 2D spatial image planes into a unified 3D physical forward-modeling framework. The same can be said about how we handle the kinematic mapping of the source galaxy. `Powerbin` [@powerbin] has already a sophifisticated and scientifically-proven [@sheu2026] methodology in discretizing spaxel regions based on their signal-to-noise. As such, we utilize `Powerbin` for our source kinematic reconstruction rather than reinvent the wheel.

Building off our principle of modularity, `IFUL` enforces a strict separation of concerns across its pipeline. Datacube ingestion, spaxel continuum background subtraction, spatial masking, and noise estimation are encapsulated within the `ImageSet` module, completely isolated from lensing physics. Spatial coordinate transformations and 2D ray-tracing abstractions are managed by `FlatModel`, while 3D spatial-spectral synthesis and kinematic model evaluations are orchestrated by `IFULModel`. Decoupling data preprocessing, 2D ray-tracing, physical kinematic modeling, and simulation tools allows researchers to easily integrate custom velocity profiles, alternative continuum subtraction routines, or adaptive spatial binning methods without altering the underlying likelihood or lens-solver engines.

# Research impact statement

`IFUL` has already demonstrated its scientific utility through direct application to observational datasets. In an upcoming study (Sheu et al. in prep.), `IFUL` was deployed to perform joint 3D spatial-spectral lens modeling of the "Carousel" strong gravitational lens system using high-resolution VLT/MUSE IFU observations [@carousel0; @carousel1; @carousel2]. By isolating emission line kinematics from foreground lens continuum light, `IFUL` successfully reconstructed the intrinsic line-of-sight velocity field and velocity dispersion of the background spiral galaxy at redshift $z = 1.432$. Crucially, incorporating spaxel-level spectroscopic constraints yielded tighter posterior probability distributions on macro lens mass model parameters compared to traditional 2D photometric modeling alone, while enabling robust constraints on high-redshift disk dynamics and central supermassive black hole mass limits.

Beyond individual lens system modeling, `IFUL` has been presented at international astrophysics conferences, receiving strong interest from the gravitational lensing and galaxy evolution communities. The software is currently being adopted by multiple research groups for ongoing projects, including the 3D spectroscopic modeling of lensed quasar host galaxies and lensed star-forming systems observed with *JWST*/NIRSpec.

# AI usage disclosure

Generative AI (Gemini 3.6 Flash) was used to check formatting, spelling, and grammar of the manuscript, as well as to assist in generating docstrings for the codebase. No generative AI was used in the creation of the underlying code itself.

# Acknowledgements

I thank Karl Glazebrook for the initial concept of 3D lens modeling, and Xiaosheng Huang, Felipe Urcelay, Evan Odell, Linus Upson, Sean Xu, Tesla Jeltema, Jackson O'Donnell, Aleksandar Cikota, and Tommaso Treu for their support in the construction of `IFUL`.

# References
