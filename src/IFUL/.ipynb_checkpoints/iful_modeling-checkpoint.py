import numpy as np
import scipy as sp
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.Util import param_util
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from tqdm import tqdm

from IFUL.util import *
from IFUL.image_set import *
from IFUL.flat_modeling import *

class IFULModel():
    def __init__(self, imageset, flatmodel, sourceplane_size, num_bins, num_rsersics):
        self.sourceplane_size = sourceplane_size
        self.num_bins = num_bins
        self.num_rsersics = num_rsersics
        
        imModel = class_creator.create_im_sim(
            flatmodel.init_pso_fitting_seq.multi_band_list,
            'multi-linear',
            flatmodel.init_pso_fitting_seq._updateManager.kwargs_model,
            bands_compute=None,
            linear_solver=False,
            image_likelihood_mask_list=np.array([imageset.mask]))
        self.kwargs_params = copy.deepcopy(flatmodel.init_pso_fit)
        self.kwargs_params.pop("kwargs_tracer_source", None)

        imModel.image_linear_solve(inv_bool=True, **self.kwargs_params)
        
        self.immodel = imModel._imageModel_list[0]
        self.sm = self.immodel.source_mapping

        self.get_sourceplane_img(flatmodel)

        self.voronoi_given_nbins(num_bins, 
                                 np.nanmax(self.source_fluxes)*2,  
                                 np.nansum(self.source_fluxes) / np.sum(~np.isnan(self.source_fluxes))**0.5 / 2, 
                                 flatmodel.init_pso_fit['kwargs_source'])

    def get_sourceplane_img(self, flatmodel):
        self.sourcecenter = np.array([flatmodel.init_pso_fit['kwargs_source'][0]['center_x'], flatmodel.init_pso_fit['kwargs_source'][0]['center_y']])
        center_pixel = self.sourceplane_size//2
        dict_sersic = copy.deepcopy(flatmodel.init_pso_fit['kwargs_source'][0])
        self.threshold_source_radius = dict_sersic['R_sersic']*self.num_rsersics
        
        dpix_mult = 1.
        not_valid = True
        
        while not_valid:
            dpix = dict_sersic['R_sersic']/self.sourceplane_size*2 * self.num_rsersics * dpix_mult
            
            pixel_locations = []
            values = []
            
            teste1 = dict_sersic['e1']
            teste2 = dict_sersic['e2']
            
            source_img = np.zeros((self.sourceplane_size, self.sourceplane_size))
            for x in np.arange(self.sourceplane_size):
                for y in np.arange(self.sourceplane_size):
                    centered_x, centered_y = (x-center_pixel)*dpix, (y-center_pixel)*dpix
                    
                    xval = self.sourcecenter[0] + centered_x
                    yval = self.sourcecenter[1] + centered_y
            
                    x_, y_ = param_util.transform_e1e2_product_average(
                        centered_x, centered_y, teste1, teste2, center_x=0, center_y=0)
                    d = (x_**2 + y_**2)**0.5
            
                    v = self.sm._light_model.surface_brightness(xval, yval, [dict_sersic])
            
                    if d >= dict_sersic['R_sersic']*self.num_rsersics:
                        source_img[x, y] = np.nan
                        continue
                    
                    pixel_locations += [[x, y]]
                    values += [v]
                    
                    source_img[x, y] = v
            
            pixel_locations = np.array(pixel_locations)
            values = np.array(values)
        
            not_valid = not is_border_all_nan(source_img)
            if not_valid:
                dpix_mult += 0.01

        self.pixel_locations = pixel_locations
        self.source_fluxes = values
        self.dpix = dpix

    def voronoi_given_nbins(self, target_y, low, high, kwargs_source, epsilon=1e-7):
        while (high - low) > epsilon:
            mid = low + (high - low) / 2.0
            bin_number, y_gen, x_gen, y_bar, x_bar, sn, nPixels, scale = voronoi_2d_binning(self.pixel_locations.T[1], self.pixel_locations.T[0], self.source_fluxes, 
                                                                                            np.ones(self.source_fluxes.shape), mid, pixelsize=None, plot=False, quiet=True)
            mid_y = len(y_gen)
            if mid_y == target_y:
                break 
            elif mid_y < target_y:
                high = mid
            else:
                low = mid
            
        bin_number, y_gen, x_gen, y_bar, x_bar, sn, nPixels, scale = voronoi_2d_binning(self.pixel_locations.T[1], self.pixel_locations.T[0], self.source_fluxes, 
                                                                                        np.ones(self.source_fluxes.shape), mid, pixelsize=None, plot=True, quiet=True)
        
        self.init_bin_sourceflux = self.sm._light_model.surface_brightness(self.sourcecenter[0] + (x_bar - self.sourceplane_size//2)*self.dpix, 
                                                                           self.sourcecenter[1] + (y_bar - self.sourceplane_size//2)*self.dpix, 
                                                                           kwargs_source)
        
        x_, y_ = param_util.transform_e1e2_product_average((x_gen - self.sourceplane_size//2)*self.dpix, 
                                                           (y_gen - self.sourceplane_size//2)*self.dpix, 
                                                           kwargs_source[0]['e1'], 
                                                           kwargs_source[0]['e2'], 
                                                           center_x=0, center_y=0)
        points_rot = rotate_points(np.array([x_, y_]).T, kwargs_source[0]['e1'], kwargs_source[0]['e2'])
        x_, y_ = points_rot[:, 0], points_rot[:, 1]
        
        self.x_bins = x_ / kwargs_source[0]['R_sersic']
        self.y_bins = y_ / kwargs_source[0]['R_sersic']

    def given_ra_dec_return_bin_no(self, x_source, y_source, source_params, return_dist=False):
        x_ra, y_dec = param_util.transform_e1e2_product_average(
            x_source - source_params['center_x'], y_source - source_params['center_y'], source_params['e1'], source_params['e2'], center_x=0, center_y=0
        )
        points_rot = rotate_points(np.array([x_ra, y_dec]).T, source_params['e1'], source_params['e2'])
        x_ra, y_dec = points_rot[:, 0], points_rot[:, 1]
    
        x_ra, y_dec = (x_ra, y_dec) / source_params['R_sersic']
    
        dists = (x_ra**2 + y_dec**2)**0.5 * source_params['R_sersic']
        res = find_closest_point_indices(np.array([self.x_bins, self.y_bins]).T, np.array([x_ra, y_dec]).T)
        res = np.array([r if d <= self.threshold_source_radius else np.nan for r, d in zip(res, dists)])
        
        if return_dist:
            return res, dists
        return res

    def gen_binned_lensed_whitelight(self, kwargs_source, kwargs_lens, binned_fluxes=None):
        if binned_fluxes is None:
            binned_fluxes = self.init_bin_sourceflux
            
        ra_grid, dec_grid = self.immodel.ImageNumerics.coordinates_evaluate
        
        x_source, y_source = self.sm._lens_model.ray_shooting(ra_grid, dec_grid, kwargs_lens)
        
        inds = self.given_ra_dec_return_bin_no(x_source, y_source, kwargs_source[0], kwargs_source[0]['R_sersic']*self.num_rsersics)
        source_light = np.array([0 if np.isnan(ind) else binned_fluxes[int(ind)] for ind in inds])
        
        return self.immodel.ImageNumerics.re_size_convolve(source_light, unconvolved=False)
        
    def gen_binned_unlensed_whitelight(self, kwargs_source, binned_fluxes=None):
        if binned_fluxes is None:
            binned_fluxes = self.init_bin_sourceflux
            
        source_img = np.zeros((self.sourceplane_size, self.sourceplane_size))
        center_pixel = self.sourceplane_size//2
        
        for x in tqdm(np.arange(self.sourceplane_size)):
            for y in np.arange(self.sourceplane_size):
                xval = self.sourcecenter[0] + (x-center_pixel)*self.dpix
                yval = self.sourcecenter[1] + (y-center_pixel)*self.dpix
        
                ind = self.given_ra_dec_return_bin_no([xval], [yval], kwargs_source[0])[0]
                if np.isnan(ind):
                    source_light = np.nan
                else:
                    source_light = binned_fluxes[int(ind)]
                          
                source_img[x, y] = source_light

        return source_img
