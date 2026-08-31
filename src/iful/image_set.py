"""
ImageSet class for managing IFU datacubes, continuum subtraction, masking, and noise calculation.

This module provides data handling functionality for 3D spectroscopic datacubes, including
continuum model subtraction, spatial mask management, noise level calculation, and initial 1D/2D
emission line fitting.
"""

import numpy as np
import scipy as sp
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

from .util import *


class ImageSet:
    """
    Class to hold, preprocess, mask, and analyze an IFU spectroscopic datacube.

    Attributes
    ----------
    zs : float
        Redshift of the source object.
    size : int
        Spatial dimension (pixel size along x/y).
    pixscale : float
        Pixel scale in degrees or arcseconds.
    var_datacube : numpy.ndarray or None
        Variance cube associated with the datacube (if provided).
    aux_info : dict
        Auxiliary metadata dictionary (e.g., WCS header, exposure time, PSF).
    datacube : numpy.ndarray
        3D continuum-subtracted data cube of shape (N_x, N_y, N_wave).
    wavelength : numpy.ndarray
        1D wavelength array corresponding to the spectral axis of `datacube`.
    datacube_whitelight : numpy.ndarray
        2D white-light image collapse (median over wavelength axis).
    mask : numpy.ndarray
        2D binary spatial mask (1 = unmasked, 0 = masked).
    mask_3d : numpy.ndarray
        3D binary mask broadcast to match `datacube` shape.
    brms_2d : float
        Background RMS noise estimated from 2D white light image.
    brms_3d : float
        Background RMS noise estimated across 3D datacube spaxels.
    img_locations : numpy.ndarray
        Array of (x, y) coordinates marking lensed image positions.
    out_mask : numpy.ndarray
        2D outlier mask computed via IQR thresholding.
    aperature_spec : numpy.ndarray
        1D aperture-extracted spectrum integrated over unmasked region.
    init_spec_fit : numpy.ndarray
        Best-fit parameters from initial spectral template fit.
    restwave_peaks : list of float
        Rest-frame emission line peak wavelengths (Angstroms).
    """

    def __init__(self, datacube, wavelengths, zs, pixscale, gap, spectra_background, var_datacube=None):
        """
        Initialize an ImageSet object and perform continuum subtraction.

        Parameters
        ----------
        datacube : numpy.ndarray
            Input 3D spectroscopic datacube of shape (N_x, N_y, N_wave).
        wavelengths : numpy.ndarray
            1D array of observed wavelengths (Angstroms).
        zs : float
            Redshift of the source.
        pixscale : float
            Pixel scale.
        gap : int
            Buffer pixel count before and after spectral background regions.
        spectra_background : int
            Number of spectral channels at start and end used to measure continuum slope.
        var_datacube : numpy.ndarray, optional
            Variance datacube matching `datacube` shape. Default is None.
        """
        self.zs = zs
        self.size = datacube.shape[0]
        self.pixscale = pixscale
        # self.wavelength_interval = wavelengths[1] - wavelengths[0]
        self.var_datacube = None
        self.continuum_subtraction(datacube, wavelengths, gap, spectra_background, var_datacube)
        self.aux_info = {}

    def continuum_subtraction(self, datacube, wavelengths, gap, spectra_background, var_datacube=None):
        """
        Estimate and subtract a spaxel-by-spaxel linear spectral continuum background.

        Parameters
        ----------
        datacube : numpy.ndarray
            Input 3D spectroscopic datacube.
        wavelengths : numpy.ndarray
            1D observed wavelength array.
        gap : int
            Gap size (number of spectral channels) to ignore between background and line region.
        spectra_background : int
            Number of spectral channels at the ends of the wavelength axis to sample continuum.
        var_datacube : numpy.ndarray, optional
            Variance cube matching input `datacube`.
        """
        buffer = gap + spectra_background
        # Estimate median flux at blue end of spectrum for each spaxel
        y1 = np.median(datacube[:, :, :spectra_background], axis=2)
        x1 = np.ones(y1.shape) * np.mean(wavelengths[:spectra_background])

        # Estimate median flux at red end of spectrum for each spaxel
        y2 = np.median(datacube[:, :, -1 * spectra_background :], axis=2)
        x2 = np.ones(y2.shape) * np.mean(wavelengths[-1 * spectra_background :])

        # Linear slope m = (y2 - y1) / (x2 - x1) per spaxel
        m = (y2 - y1) / (x2 - x1)
        continuum = np.zeros(datacube.shape)
        for x in np.arange(continuum.shape[0]):
            for y in np.arange(continuum.shape[1]):
                continuum_spax = m[x, y] * wavelengths - m[x, y] * x1[x, y] + y1[x, y]
                continuum[x, y, :] = continuum_spax
        datacube = datacube - continuum

        # Truncate datacube along wavelength axis to discard background buffer regions
        trunc_inds = np.array(
            [
                i
                for i, w in enumerate(wavelengths)
                if i >= buffer and i < len(wavelengths) - buffer
            ]
        )

        self.datacube = datacube[:, :, np.min(trunc_inds) : np.max(trunc_inds) + 1]
        if var_datacube is not None:
            self.var_datacube = var_datacube[:, :, np.min(trunc_inds) : np.max(trunc_inds) + 1]
        else:
            self.var_datacube = None
        self.wavelength = wavelengths[np.min(trunc_inds) : np.max(trunc_inds) + 1]

        # Generate 2D white light collapsed image
        self.datacube_whitelight = np.nanmedian(self.datacube, axis=2)
        self.mask = np.ones(self.datacube_whitelight.shape)

        # Broadcast 2D mask to 3D cube shape
        mask_3d = np.array([self.mask for _ in np.arange(self.datacube.shape[-1])])
        self.mask_3d = np.moveaxis(mask_3d, [0], [2])

    def noise_level_set(
        self, mask_img_size, vminmax=[None, None], additional_mask=None
    ):
        """
        Estimate 2D and 3D background RMS noise levels after masking out central source images.

        Parameters
        ----------
        mask_img_size : float
            Radius of circular mask (in pixels) placed around each lensed image position.
        vminmax : list of float, optional
            Display bounds `[vmin, vmax]` for imshow visualization plot.
        additional_mask : numpy.ndarray, optional
            Additional 2D binary mask array to apply.
        """
        img_mask = np.ones(self.datacube_whitelight.shape)
        for l in self.img_locations:
            img_mask *= mask_circle(l[0], l[1], mask_img_size, img_mask.shape)

        if additional_mask is not None:
            img_mask *= additional_mask

        img_mask_3d = np.array([img_mask for _ in np.arange(self.datacube.shape[-1])])
        img_mask_3d = np.moveaxis(img_mask_3d, [0], [2])

        bkg_std_cutout3d_img = self.datacube * self.mask_3d * img_mask_3d

        bkg_std_cutout3d_img_nan = copy.deepcopy(bkg_std_cutout3d_img)
        bkg_std_cutout3d_img_nan[bkg_std_cutout3d_img_nan == 0.0] = np.nan
        median_bkg_spec = np.nanmedian(bkg_std_cutout3d_img_nan, axis=[0, 1])
        median_bkg_spec_3d = np.broadcast_to(
            median_bkg_spec,
            (self.datacube.shape[0], self.datacube.shape[1], len(median_bkg_spec)),
        )

        bkg_std_cutout3d_img -= median_bkg_spec_3d
        self.datacube -= median_bkg_spec_3d
        self.datacube_whitelight = np.nanmedian(self.datacube, axis=2)

        # Compute 3D RMS noise level
        bkg_std_cutout3d = bkg_std_cutout3d_img.reshape(-1)
        bkg_std_cutout3d = bkg_std_cutout3d[~np.isnan(bkg_std_cutout3d)]
        self.brms_3d = (np.nansum(bkg_std_cutout3d**2) / len(bkg_std_cutout3d)) ** 0.5

        # Compute 2D RMS noise level
        bkg_std_cutout_img = self.datacube_whitelight * self.mask * img_mask
        bkg_std_cutout = bkg_std_cutout_img.reshape(-1)
        bkg_std_cutout = bkg_std_cutout[~np.isnan(bkg_std_cutout)]
        self.brms_2d = (np.sum(bkg_std_cutout**2) / len(bkg_std_cutout)) ** 0.5

        pltimage = bkg_std_cutout_img
        pltimage -= np.nanmedian(pltimage)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.rot90(pltimage.T, 3), vmin=vminmax[0], vmax=vminmax[1])

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()

    def mask_outliers(self, scale_l=5, scale_u=100, vminmax=[None, None]):
        """
        Identify and mask spatial intensity outliers in the datacube using IQR clipping.

        Parameters
        ----------
        scale_l : float, default=5
            Lower IQR multiplier threshold.
        scale_u : float, default=100
            Upper IQR multiplier threshold.
        vminmax : list of float, optional
            Display bounds `[vmin, vmax]` for visualization plot.
        """
        t_whiteimage = np.nansum(self.datacube, axis=2)

        out_mask = get_outlier_mask_iqr(
            t_whiteimage, scale_l=scale_l, scale_u=scale_u
        ).astype(float)
        out_mask *= (~np.isnan(np.sum(self.datacube, axis=2))).astype(float)
        self.out_mask = out_mask
        self.mask = out_mask

        pltimage = self.datacube_whitelight * self.out_mask
        pltimage -= np.nanmedian(pltimage)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.rot90(pltimage.T, 3), vmin=vminmax[0], vmax=vminmax[1])

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()

    def mark_image_locations(self, locations_plt):
        """
        Mark and store lensed image position coordinates on the white light image.

        Parameters
        ----------
        locations_plt : array-like
            List of (x, y) coordinate pairs in display pixel orientation.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(np.rot90((self.datacube_whitelight * self.mask).T, 3))

        locations = np.array(
            [
                [
                    j,  # -(j - (self.size / 2 - 0.5)) + (self.size / 2 - 0.5),
                    -(i - (self.size / 2 - 0.5)) + (self.size / 2 - 0.5),
                ]
                for i, j in locations_plt
            ]
        )
        plt.scatter(locations_plt.T[0], locations_plt.T[1], c="r", s=5)
        self.img_locations = locations

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()

    def set_mask(self, mask):
        """
        Apply a custom 2D binary spatial mask combined with the outlier mask.

        Parameters
        ----------
        mask : numpy.ndarray
            2D binary array (1 for valid pixels, 0 for masked pixels).
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(np.rot90((self.datacube_whitelight * mask * self.out_mask).T, 3))
        self.mask = mask * self.out_mask

        mask_3d = np.array([self.mask for _ in np.arange(self.datacube.shape[-1])])
        self.mask_3d = np.moveaxis(mask_3d, [0], [2])

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()

    def get_initial_spec_fit_mask(self, img_size, vminmax=[None, None]):
        """
        Extract background aperture spectrum by masking out image regions of given radius.

        Parameters
        ----------
        img_size : float
            Radius around lensed images to exclude (in pixels).
        vminmax : list of float, optional
            Display bounds `[vmin, vmax]` for visualization plot.
        """
        img_mask = np.ones(self.datacube_whitelight.shape)
        for l in self.img_locations:
            img_mask *= mask_circle(l[0], l[1], img_size, img_mask.shape)
        img_mask = (~img_mask.astype(bool)).astype(float)

        img_mask_3d = np.array([img_mask for _ in np.arange(self.datacube.shape[-1])])
        img_mask_3d = np.moveaxis(img_mask_3d, [0], [2])

        self.aperature_spec = np.nansum(
            np.nansum((self.datacube * img_mask_3d * self.mask_3d), axis=0), axis=0
        )

        pltimage = self.datacube_whitelight * self.mask * img_mask
        pltimage -= np.nanmedian(pltimage)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.rot90((pltimage).T, 3), vmin=vminmax[0], vmax=vminmax[1])

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()

    def get_initial_spec_fit(
        self, restwave_peaks, init_spec_params, rewrite_zs=True, adjust_slope=False
    ):
        """
        Fit emission line template model to extracted aperture spectrum.

        Parameters
        ----------
        restwave_peaks : list of float
            Rest-frame peak wavelengths for emission lines (Angstroms).
        init_spec_params : list of float
            Initial parameters `[z, sigma_ang, amp_0, ratio_1, ratio_2, ...]`.
        rewrite_zs : bool, default=True
            If True, update `self.zs` attribute with best-fit redshift.
        adjust_slope : bool, default=False
            If True, include linear slope and baseline offset parameters in template.
        """
        self.restwave_peaks = restwave_peaks

        init_spec = self.gen_2d_spec(init_spec_params, adjust_slope)

        def min_fnc(x):
            return np.sum(
                (self.aperature_spec - (self.gen_2d_spec(x, adjust_slope))) ** 2
            )

        self.init_spec_fit = minimize(min_fnc, init_spec_params).x

        plt.plot(self.wavelength, self.aperature_spec, c="blue", label="aperture data")
        plt.plot(self.wavelength, init_spec, ls="--", c="red", label="initial fit")
        plt.plot(
            self.wavelength,
            self.gen_2d_spec(self.init_spec_fit),
            c="orange",
            label="aperture fit",
        )
        plt.xlabel("Observed wavelength (Angstrom)")
        plt.legend()

        if rewrite_zs:
            self.zs = self.init_spec_fit[0]
        print(self.init_spec_fit)

    def gen_2d_spec(self, params, slope=False, wavelength_range=None):
        """
        Generate 1D emission line spectral model evaluated across wavelength array.

        Parameters
        ----------
        params : list or array-like
            Parameters formatted as `[z, sigma_ang, amp_0, ratio_1, ..., (m, b)]`.
        slope : bool, default=False
            Whether linear baseline slope `m` and intercept `b` are included in params.
        wavelength_range : numpy.ndarray, optional
            Custom 1D wavelength array to evaluate model on. Defaults to `self.wavelength`.

        Returns
        -------
        numpy.ndarray
            Modeled flux array across `wavelength_range`.
        """
        # params_format: z [1], sigma [1], amp_0 [1], ratios [self.restwave_peaks - 1]
        z, sigma_ang, amp_0, ratios = params[0], params[1], params[2], params[3:]
        if slope:
            m = params[-2]
            b = params[-1]
            ratios = ratios[:-2]
        else:
            m = 0
            b = 0

        if wavelength_range is None:
            wavelength_range = self.wavelength

        ratios_f = [1] + list(np.array(ratios))
        return np.array(
            [
                np.sum(
                    [
                        norm_dist(
                            amp_0 * ratio,
                            rp * (1 + z),
                            sigma_ang, #/ (self.wavelength[1] - self.wavelength[0]),
                            w,
                        )
                        for rp, ratio in zip(self.restwave_peaks, ratios_f)
                    ]
                )
                + (w - np.mean(self.wavelength)) * m
                + b
                for w in wavelength_range
            ]
        )

    def gen_2d_spec_fixratios(self, params):
        """
        Generate spectral model fixing line flux ratios to `self.init_spec_fit`.

        Parameters
        ----------
        params : list or array-like
            Parameters `[z, sigma_ang, amp_0]`.

        Returns
        -------
        numpy.ndarray
            Modeled flux array evaluated over `self.wavelength`.
        """
        # params_format: z [1], sigma [1], amp_0 [1], ratios [self.restwave_peaks - 1]
        z, sigma_ang, amp_0 = params[0], params[1], params[2]

        ratios_f = [1] + list(self.init_spec_fit[3:])  # list(np.array(ratios))
        return np.array(
            [
                np.sum(
                    [
                        norm_dist(
                            amp_0 * ratio,
                            rp * (1 + z),
                            sigma_ang,# / (self.wavelength[1] - self.wavelength[0]),
                            w,
                        )
                        for rp, ratio in zip(self.restwave_peaks, ratios_f)
                    ]
                )
                for w in self.wavelength
            ]
        )

    def add_aux_info(self, add_aux):
        """
        Merge additional dictionary items into `aux_info`.

        Parameters
        ----------
        add_aux : dict
            Dictionary of metadata entries to add or update.
        """
        self.aux_info = {**self.aux_info, **add_aux}

