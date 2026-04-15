import numpy as np
import scipy as sp
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

from IFUL.util import *

class ImageSet():
    def __init__(self, datacube, var_datacube, wavelengths, zs, pixscale, gap, spectra_background):
        self.zs = zs
        self.size = datacube.shape[0]
        self.pixscale = pixscale
        # self.wavelength_interval = wavelengths[1] - wavelengths[0]
        self.continuum_subtraction(datacube, var_datacube, wavelengths, gap, spectra_background)
        self.aux_info = {}

    def continuum_subtraction(self, datacube, var_datacube, wavelengths, gap, spectra_background):
        buffer = gap + spectra_background
        y1 = np.median(datacube[:, :, :spectra_background], axis=2)
        x1 = np.ones(y1.shape)*np.mean(wavelengths[:spectra_background])
        
        y2 = np.median(datacube[:, :, -1*spectra_background:], axis=2)
        x2 = np.ones(y2.shape)*np.mean(wavelengths[-1*spectra_background:])
        
        m = (y2 - y1) / (x2 - x1)
        continuum = np.zeros(datacube.shape)
        for x in np.arange(continuum.shape[0]):
            for y in np.arange(continuum.shape[1]):
                continuum_spax = m[x, y]*wavelengths - m[x, y]*x1[x, y] + y1[x, y]
                continuum[x, y, :] = continuum_spax
        datacube = datacube - continuum
        
        trunc_inds = np.array([i for i, w in enumerate(wavelengths) if i >= buffer and i < len(wavelengths)-buffer])
        
        self.datacube = datacube[:, :, np.min(trunc_inds):np.max(trunc_inds)+1]
        self.var_datacube = var_datacube[:, :, np.min(trunc_inds):np.max(trunc_inds)+1]
        self.wavelength = wavelengths[np.min(trunc_inds):np.max(trunc_inds)+1]
        
        self.datacube_whitelight = np.nanmedian(self.datacube, axis=2)
        self.mask = np.ones(self.datacube_whitelight.shape)

        mask_3d = np.array([self.mask for _ in np.arange(self.datacube.shape[-1])])
        self.mask_3d = np.moveaxis(mask_3d, [0], [2])

    def noise_level_set(self, mask_img_size, vminmax=[None, None], additional_mask=None):
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
        median_bkg_spec_3d = np.broadcast_to(median_bkg_spec, (self.datacube.shape[0], self.datacube.shape[1], len(median_bkg_spec)))
        
        bkg_std_cutout3d_img -= median_bkg_spec_3d        
        self.datacube -= median_bkg_spec_3d
        self.datacube_whitelight = np.nanmedian(self.datacube, axis=2)
        
        bkg_std_cutout3d = bkg_std_cutout3d_img.reshape(-1)
        bkg_std_cutout3d = bkg_std_cutout3d[~np.isnan(bkg_std_cutout3d)]
        self.brms_3d = (np.nansum(bkg_std_cutout3d**2)/len(bkg_std_cutout3d))**0.5
        
        bkg_std_cutout_img = self.datacube_whitelight * self.mask * img_mask
        bkg_std_cutout = bkg_std_cutout_img.reshape(-1)
        bkg_std_cutout = bkg_std_cutout[~np.isnan(bkg_std_cutout)]
        self.brms_2d = (np.sum(bkg_std_cutout**2)/len(bkg_std_cutout))**0.5

        pltimage = bkg_std_cutout_img
        pltimage -= np.nanmedian(pltimage)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.rot90(pltimage.T, 3), vmin=vminmax[0], vmax=vminmax[1])
        
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()

    def mask_outliers(self, scale_l=5, scale_u=100, vminmax=[None, None]):
        t_whiteimage = np.nansum(self.datacube, axis=2)

        out_mask = get_outlier_mask_iqr(t_whiteimage, scale_l=scale_l, scale_u=scale_u).astype(float)
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

    def mark_image_locations(self, locations):
        plt.figure(figsize=(8, 8))
        plt.imshow(np.rot90((self.datacube_whitelight * self.mask).T, 3))

        locations_plt = np.array([[-(j - (self.size/2-0.5)) + (self.size/2-0.5), (i - (self.size/2-0.5)) + (self.size/2-0.5)] for i, j in locations])
        plt.scatter(locations_plt.T[0], locations_plt.T[1], c='r', s=5)
        self.img_locations = locations

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()

    def set_mask(self, mask):
        plt.figure(figsize=(8, 8))
        plt.imshow(np.rot90((self.datacube_whitelight * mask * self.out_mask).T, 3))
        self.mask = mask * self.out_mask
        
        mask_3d = np.array([self.mask for _ in np.arange(self.datacube.shape[-1])])
        self.mask_3d = np.moveaxis(mask_3d, [0], [2])

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()

    def get_initial_spec_fit_mask(self, img_size, vminmax=[None, None]):
        img_mask = np.ones(self.datacube_whitelight.shape)
        for l in self.img_locations:
            img_mask *= mask_circle(l[0], l[1], img_size, img_mask.shape)
        img_mask = (~img_mask.astype(bool)).astype(float)
        
        img_mask_3d = np.array([img_mask for _ in np.arange(self.datacube.shape[-1])])
        img_mask_3d = np.moveaxis(img_mask_3d, [0], [2])

        self.aperature_spec = np.nansum(np.nansum((self.datacube * img_mask_3d * self.mask_3d), axis=0), axis=0)

        pltimage = self.datacube_whitelight * self.mask * img_mask
        pltimage -= np.nanmedian(pltimage)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.rot90((pltimage).T, 3), vmin=vminmax[0], vmax=vminmax[1])

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.show()

    def get_initial_spec_fit(self, restwave_peaks, init_spec_params, rewrite_zs=True, adjust_slope=False):
        self.restwave_peaks = restwave_peaks

        init_spec = self.gen_2d_spec(init_spec_params, adjust_slope)
        def min_fnc(x):
            return np.sum((self.aperature_spec - (self.gen_2d_spec(x, adjust_slope)))**2)
        self.init_spec_fit = minimize(min_fnc, init_spec_params).x
        
        plt.plot(self.wavelength, self.aperature_spec, c='blue', label='aperture data')
        plt.plot(self.wavelength, init_spec, ls='--', c='red', label='initial fit')
        plt.plot(self.wavelength, self.gen_2d_spec(self.init_spec_fit), c='orange', label='aperture fit')
        plt.xlabel("Observed wavelength (Angstrom)")
        plt.legend()

        if rewrite_zs:
            self.zs = self.init_spec_fit[0]
        print(self.init_spec_fit)

    def gen_2d_spec(self, params, slope=False):
        # params_format: z [1], sigma [1], amp_0 [1], ratios [self.restwave_peaks - 1]
        z, sigma_ang, amp_0, ratios = params[0], params[1], params[2], params[3:]
        if slope:
            m = params[-2]
            b = params[-1]
            ratios = ratios[:-2]
        else:
            m = 0
            b = 0
        
        ratios_f = [1] + list(np.array(ratios))
        return np.array([np.sum([norm_dist(amp_0 * ratio, rp*(1+z), sigma_ang/(self.wavelength[1] - self.wavelength[0]), w) 
                                 for rp, ratio in zip(self.restwave_peaks, ratios_f)]) + (w - np.mean(self.wavelength))*m + b 
                         for w in self.wavelength])

    def gen_2d_spec_fixratios(self, params):
        # params_format: z [1], sigma [1], amp_0 [1], ratios [self.restwave_peaks - 1]
        z, sigma_ang, amp_0 = params[0], params[1], params[2]
        
        ratios_f = [1] + list(self.init_spec_fit[3:]) #list(np.array(ratios))
        return np.array([np.sum([norm_dist(amp_0 * ratio, rp*(1+z), sigma_ang/(self.wavelength[1] - self.wavelength[0]), w) 
                                 for rp, ratio in zip(self.restwave_peaks, ratios_f)]) 
                         for w in self.wavelength])
    
    def add_aux_info(self, add_aux):
        self.aux_info = {**self.aux_info, **add_aux}