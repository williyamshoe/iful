import numpy as np
import scipy as sp
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from astropy.wcs import WCS

from IFUL.util import *
from IFUL.image_set import *

class FlatModel():
    def __init__(self, imageset, lensmodel, sourcemodel):
        # self.imageset = imageset
        self.lensmodel = lensmodel
        self.sourcemodel = sourcemodel

        self.init_pos_fit = None
        self.init_pso_fit = None
        self.init_pso_fitting_seq = None
        self.mcmc_chains = None

        self.header_wcs = imageset.aux_info['header_wcs']

        self.make_lenstronomy_params(imageset)
        self.init_lens_centers = self.convert_pixel_to_ra_dec(imageset.aux_info['init_lens_center'])
        self.init_img_centers = self.convert_pixel_to_ra_dec(imageset.img_locations)
        # self.data_class = ImageData(**self.multi_band_list[0])

    def convert_pixel_to_ra_dec(self, locations):
        ras, decs = [], []
        for iloc in locations:
            world = WCS(self.header_wcs).pixel_to_world(iloc[0], iloc[1], 0)
            ra = 360-world[0].dec.deg
            dec = 360-world[0].ra.deg
        
            if ra > 180:
                ra -= 360
            if dec > 180:
                dec -= 360
        
            ras += [ra*3600]
            decs += [dec*3600]
        return np.array([ras, decs]).T

    def make_lenstronomy_params(self, imageset):
        world = WCS(self.header_wcs).pixel_to_world(0, 0, 0)
        ra_at_xy_0 = 360-world[0].dec.deg 
        dec_at_xy_0 = 360-world[0].ra.deg 
        if ra_at_xy_0 > 180:
            ra_at_xy_0 -= 360
        if dec_at_xy_0 > 180:
            dec_at_xy_0 -= 360
        ra_at_xy_0 *= 3600
        dec_at_xy_0 *= 3600

        transform_pix2angle = np.array([[-1*imageset.aux_info['pixscale']*3600, 0.], [0., imageset.aux_info['pixscale']*3600]])
        # self.init_psf_fwhm = calculate_fwhm(imageset.aux_info['psf_info']['amps'], self.aux_info['psf_info']['sigmas'])

        kwargs_datas = {'image_data': copy.deepcopy(imageset.datacube_whitelight),
                       'background_rms': copy.deepcopy(imageset.brms_2d),
                       'noise_map': None,
                       'exposure_time': copy.deepcopy(imageset.aux_info['exptime']),
                       'ra_at_xy_0': copy.deepcopy(ra_at_xy_0),
                       'dec_at_xy_0': copy.deepcopy(dec_at_xy_0), 
                       'transform_pix2angle': copy.deepcopy(transform_pix2angle),
                       }
        kwargs_psf = {'psf_type': 'PIXEL', 
                  'kernel_point_source': copy.deepcopy(imageset.aux_info['psf_info']['final_psf']),
                  'kernel_point_source_init': copy.deepcopy(imageset.aux_info['psf_info']['final_psf']),
                  'psf_variance_map': copy.deepcopy(np.ones(imageset.aux_info['psf_info']['final_psf'].shape)*1e-7),
                  }
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
        self.multi_band_list = [kwargs_datas, kwargs_psf, kwargs_numerics]