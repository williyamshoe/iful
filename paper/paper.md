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



# State of the field                                                                                                                  



# Software design



# Research impact statement



# AI usage disclosure



# Acknowledgements



# References
