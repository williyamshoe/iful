"""
Flat 2D lens and source modeling interface using Lenstronomy.

This module provides the `FlatModel` class, which initializes 2D photometric modeling parameters
and sets up coordinate transformations and Lenstronomy data lists for lensed system fitting.
"""

import numpy as np
import scipy as sp
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from astropy.wcs import WCS

from .util import *
from .image_set import *


class FlatModel:
    """
    2D photometric modeling class interfacing with Lenstronomy.

    Attributes
    ----------
    lensmodel : list of str
        List of Lenstronomy lens model profile names.
    sourcemodel : list of str
        List of Lenstronomy source light model profile names.
    init_pos_fit : optional
        Results of initial position fitting.
    init_fitting_seq : Lenstronomy FittingSequence or None
        Active fitting sequence instance for optimization.
    init_pso_fitting_seq : optional
        Particle Swarm Optimization fitting sequence.
    init_fit : optional
        Initial optimization fit results.
    init_pso_fit : dict or None
        Dictionary containing PSO best-fit kwargs parameters.
    mcmc_chains : numpy.ndarray or None
        MCMC sampling chains.
    header_wcs : astropy.io.fits.Header
        WCS header dictionary from auxiliary info.
    init_lens_centers : numpy.ndarray
        Lens center coordinates converted from pixel to RA/Dec offset (arcsec).
    init_img_centers : numpy.ndarray
        Image position coordinates converted from pixel to RA/Dec offset (arcsec).
    multi_band_list : list
        Lenstronomy multi-band list `[kwargs_data, kwargs_psf, kwargs_numerics]`.
    """

    def __init__(self, imageset, lensmodel, sourcemodel):
        """
        Initialize the FlatModel.

        Parameters
        ----------
        imageset : ImageSet
            ImageSet instance containing data cube and aux_info.
        lensmodel : list of str
            List of lens model names (e.g. ['EPL_Q_PHI', 'SHEAR']).
        sourcemodel : list of str
            List of source light model names (e.g. ['SERSIC_ELLIPSE']).
        """
        # self.imageset = imageset
        self.lensmodel = lensmodel
        self.sourcemodel = sourcemodel

        # Initialize fitting result placeholders
        self.init_pos_fit = None
        self.init_fitting_seq = None
        self.init_pso_fitting_seq = None   
        self.init_fit = None
        self.init_pso_fit = None
        
        self.mcmc_chains = None

        # Extract WCS header from auxiliary information
        self.header_wcs = imageset.aux_info["header_wcs"]

        # Construct Lenstronomy data configuration and coordinate transforms
        self.make_lenstronomy_params(imageset)
        self.init_lens_centers = self.convert_pixel_to_ra_dec(
            imageset.aux_info["init_lens_center"]
        )
        self.init_img_centers = self.convert_pixel_to_ra_dec(imageset.img_locations)
        # self.data_class = ImageData(**self.multi_band_list[0])

    def convert_pixel_to_ra_dec(self, locations):
        """
        Convert pixel coordinates to RA and Dec relative sky offsets (in arcseconds).

        Parameters
        ----------
        locations : array-like
            List or array of (x, y) pixel coordinates.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 2) containing (RA, Dec) offsets in arcseconds.
        """
        ras, decs = [], []
        for iloc in locations:
            # Convert pixel (x, y) to sky coordinates using Astropy WCS
            world = WCS(self.header_wcs).pixel_to_world(iloc[0], iloc[1], 0)
            ra = 360 - world[0].dec.deg
            dec = 360 - world[0].ra.deg

            if ra > 180:
                ra -= 360
            if dec > 180:
                dec -= 360

            ras += [ra * 3600]
            decs += [dec * 3600]
        return np.array([ras, decs]).T

    def make_lenstronomy_params(self, imageset):
        """
        Build kwargs_data, kwargs_psf, and kwargs_numerics for Lenstronomy fitting.

        Parameters
        ----------
        imageset : ImageSet
            ImageSet instance providing whitelight image, noise level, pixel scale, and PSF.
        """
        # Determine sky reference coordinate at (x=0, y=0)
        world = WCS(self.header_wcs).pixel_to_world(0, 0, 0)
        ra_at_xy_0 = 360 - world[0].dec.deg
        dec_at_xy_0 = 360 - world[0].ra.deg
        if ra_at_xy_0 > 180:
            ra_at_xy_0 -= 360
        if dec_at_xy_0 > 180:
            dec_at_xy_0 -= 360
        ra_at_xy_0 *= 3600
        dec_at_xy_0 *= 3600

        # Construct transformation matrix from pixel coordinates to angle (arcsec)
        transform_pix2angle = np.array(
            [
                [-1 * imageset.aux_info["pixscale"] * 3600, 0.0],
                [0.0, imageset.aux_info["pixscale"] * 3600],
            ]
        )
        # self.init_psf_fwhm = calculate_fwhm(imageset.aux_info['psf_info']['amps'], self.aux_info['psf_info']['sigmas'])

        # Data kwargs for Lenstronomy
        kwargs_datas = {
            "image_data": copy.deepcopy(imageset.datacube_whitelight),
            "background_rms": copy.deepcopy(imageset.brms_2d),
            "noise_map": None,
            "exposure_time": copy.deepcopy(imageset.aux_info["exptime"]),
            "ra_at_xy_0": copy.deepcopy(ra_at_xy_0),
            "dec_at_xy_0": copy.deepcopy(dec_at_xy_0),
            "transform_pix2angle": copy.deepcopy(transform_pix2angle),
        }
        # PSF kwargs for Lenstronomy
        kwargs_psf = {
            "psf_type": "PIXEL",
            "kernel_point_source": copy.deepcopy(imageset.aux_info["final_psf"]),
            "kernel_point_source_init": copy.deepcopy(imageset.aux_info["final_psf"]),
            "psf_variance_map": copy.deepcopy(
                np.ones(imageset.aux_info["final_psf"].shape) * 1e-7
            ),
        }
        # Numerics kwargs for Lenstronomy
        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
        }
        # Store as standard Lenstronomy multi-band list format
        self.multi_band_list = [kwargs_datas, kwargs_psf, kwargs_numerics]

