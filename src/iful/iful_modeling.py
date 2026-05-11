import numpy as np
import scipy as sp
import copy
import gc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
from scipy.optimize import minimize
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.Util import param_util
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from tqdm import tqdm
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from .util import *
from .image_set import *
from .flat_modeling import *


class IFULModel:
    def __init__(
        self,
        imageset,
        flatmodel,
        iful_profiles,
        sourceplane_size,
        num_bins,
        num_rsersics,
        spectral_res,
        equal_weight_voronoi=False,
        constant_val=0.0,
        d_s=None,
    ):
        self.imset = imageset
        self.sourceplane_size = sourceplane_size
        self.num_bins = num_bins
        self.num_rsersics = num_rsersics
        self.init_fitting_seq = flatmodel.init_fitting_seq
        self.spectral_res = spectral_res
        self.constant_val = constant_val
        self.iful_profiles = iful_profiles
        self.equal_weight_voronoi = equal_weight_voronoi

        # Angular diameter distance D_s only used if using a BH profile
        if d_s is None:
            self.d_s = FlatLambdaCDM(H0=70, Om0=0.3).angular_diameter_distance(imset4.zs).to(u.kpc).value
        else:
            self.d_s = d_s

        if "SERSIC" not in iful_profiles:
            self.init_fitting_seq.fit_sequence(
                [
                    ["update_settings", {"source_add_fixed": [[0, ["n_sersic"]]]}],
                ]
            )

        self.imModel_classcreator = class_creator.create_im_sim(
            self.init_fitting_seq.multi_band_list,
            "multi-linear",
            self.init_fitting_seq._updateManager.kwargs_model,
            bands_compute=None,
            linear_solver=False,
            image_likelihood_mask_list=np.array([imageset.mask]),
        )
        kwargs_params = copy.deepcopy(flatmodel.init_pso_fit)
        kwargs_params.pop("kwargs_tracer_source", None)

        self.immodel_init = copy.deepcopy(self.imModel_classcreator)
        self.immodel_init.image_linear_solve(inv_bool=True, **kwargs_params)

        # immodel_init = immodel_init._imageModel_list[0]
        self.sm_init = self.immodel_init._imageModel_list[0].source_mapping

        self.get_sourceplane_img(flatmodel)

        if np.sum(["VORONOI" in s for s in self.iful_profiles]) >= 1 and self.equal_weight_voronoi:
            source_fluxes_arg = copy.deepcopy(self.source_fluxes)
            source_fluxes_arg[~np.isnan(source_fluxes_arg)] = 1.
            self.voronoi_given_nbins(
                num_bins,
                np.nanmax(source_fluxes_arg) * 2,
                np.nansum(source_fluxes_arg) / np.sum(~np.isnan(source_fluxes_arg)) ** 0.5 / 2,
                flatmodel.init_pso_fit["kwargs_source"],
                source_fluxes_arg
            )
        elif np.sum(["VORONOI" in s for s in self.iful_profiles]) >= 1:
            self.voronoi_given_nbins(
                num_bins,
                np.nanmax(self.source_fluxes) * 2,
                np.nansum(self.source_fluxes) / np.sum(~np.isnan(self.source_fluxes)) ** 0.5 / 2,
                flatmodel.init_pso_fit["kwargs_source"],
                self.source_fluxes
            )
        else:
            self.num_bins = 0

        self.init_sersic_amp = flatmodel.init_pso_fit["kwargs_source"][0]["amp"]

        self.len_model_numparams = self.init_fitting_seq.param_class.num_param()[0]
        self.v_los_fnc, self.v_los_numparams = self.decide_profiles_fnc(
            iful_profiles[0], self.num_bins
        )
        self.v_disp_fnc, self.v_disp_numparams = self.decide_profiles_fnc(
            iful_profiles[1], self.num_bins
        )
        self.flx_fnc, self.flx_numparams = self.decide_profiles_fnc(
            iful_profiles[2], self.num_bins
        )

        self.obs_datacube = np.transpose(self.imset.datacube, (2, 0, 1))
        # self.var_datacube = np.transpose(self.imset.var_datacube, (2, 0, 1))
        self.datacube_mask = np.transpose(self.imset.mask_3d, (2, 0, 1))
        self.datacube_unc = self.imset.brms_3d
        self.central_wave = np.mean(self.imset.wavelength)

        self.init_lenstronomy_args = self.init_fitting_seq.param_class.kwargs2args(
            **flatmodel.init_pso_fit
        )

        ra_grid, dec_grid = self.immodel_init._imageModel_list[
            0
        ].ImageNumerics.coordinates_evaluate
        self.init_x_source_vals, self.init_y_source_vals = (
            self.sm_init._lens_model.ray_shooting(
                ra_grid, dec_grid, kwargs_params["kwargs_lens"]
            )
        )

    def get_num_free_params(self, linear_solve=False):
        num_params = (
            self.len_model_numparams
            + self.v_los_numparams
            + self.v_disp_numparams
        )
        # If we are linearly solving for flux, they aren't free parameters in the optimizer
        if not linear_solve:
            num_params += self.flx_numparams
        return num_params

    def get_sourceplane_img(self, flatmodel):
        self.sourcecenter = np.array(
            [
                flatmodel.init_pso_fit["kwargs_source"][0]["center_x"],
                flatmodel.init_pso_fit["kwargs_source"][0]["center_y"],
            ]
        )
        center_pixel = self.sourceplane_size // 2
        dict_sersic = copy.deepcopy(flatmodel.init_pso_fit["kwargs_source"][0])

        dpix_mult = 1.0
        not_valid = True

        while not_valid:
            dpix = (
                dict_sersic["R_sersic"]
                / self.sourceplane_size
                * 2
                * self.num_rsersics
                * dpix_mult
            )

            pixel_locations = []
            values = []

            teste1 = dict_sersic["e1"]
            teste2 = dict_sersic["e2"]

            source_img = np.zeros((self.sourceplane_size, self.sourceplane_size))
            for x in np.arange(self.sourceplane_size):
                for y in np.arange(self.sourceplane_size):
                    centered_x, centered_y = (
                        (x - center_pixel) * dpix,
                        (y - center_pixel) * dpix,
                    )

                    xval = self.sourcecenter[0] + centered_x
                    yval = self.sourcecenter[1] + centered_y

                    x_, y_ = param_util.transform_e1e2_product_average(
                        centered_x, centered_y, teste1, teste2, center_x=0, center_y=0
                    )
                    d = (x_**2 + y_**2) ** 0.5

                    v = self.sm_init._light_model.surface_brightness(
                        xval, yval, [dict_sersic]
                    )

                    if d >= dict_sersic["R_sersic"] * self.num_rsersics:
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

    def voronoi_given_nbins(self, target_y, low, high, kwargs_source, source_fluxes_arg, epsilon=1e-7):
        while (high - low) > epsilon:
            mid = low + (high - low) / 2.0
            bin_number, y_gen, x_gen, y_bar, x_bar, sn, nPixels, scale = (
                voronoi_2d_binning(
                    self.pixel_locations.T[1],
                    self.pixel_locations.T[0],
                    source_fluxes_arg,
                    np.ones(source_fluxes_arg.shape),
                    mid,
                    pixelsize=None,
                    plot=False,
                    quiet=True,
                )
            )
            mid_y = len(y_gen)
            if mid_y == target_y:
                break
            elif mid_y < target_y:
                high = mid
            else:
                low = mid

        bin_number, y_gen, x_gen, y_bar, x_bar, sn, nPixels, scale = voronoi_2d_binning(
            self.pixel_locations.T[1],
            self.pixel_locations.T[0],
            source_fluxes_arg,
            np.ones(source_fluxes_arg.shape),
            mid,
            pixelsize=None,
            plot=True,
            quiet=True,
        )

        self.init_bin_sourceflux = self.sm_init._light_model.surface_brightness(
            self.sourcecenter[0] + (x_bar - self.sourceplane_size // 2) * self.dpix,
            self.sourcecenter[1] + (y_bar - self.sourceplane_size // 2) * self.dpix,
            kwargs_source,
        )

        x_, y_ = param_util.transform_e1e2_product_average(
            (x_gen - self.sourceplane_size // 2) * self.dpix,
            (y_gen - self.sourceplane_size // 2) * self.dpix,
            kwargs_source[0]["e1"],
            kwargs_source[0]["e2"],
            center_x=0,
            center_y=0,
        )
        points_rot = rotate_points(
            np.array([x_, y_]).T, kwargs_source[0]["e1"], kwargs_source[0]["e2"]
        )
        x_, y_ = points_rot[:, 0], points_rot[:, 1]

        self.x_bins = x_ / kwargs_source[0]["R_sersic"]
        self.y_bins = y_ / kwargs_source[0]["R_sersic"]
        self.num_bins = len(self.y_bins)

    def given_ra_dec_return_bin_no(
        self, x_source, y_source, source_params, return_dist=False
    ):
        x_ra, y_dec = param_util.transform_e1e2_product_average(
            x_source - source_params["center_x"],
            y_source - source_params["center_y"],
            source_params["e1"],
            source_params["e2"],
            center_x=0,
            center_y=0,
        )
        points_rot = rotate_points(
            np.array([x_ra, y_dec]).T, source_params["e1"], source_params["e2"]
        )
        x_ra, y_dec = points_rot[:, 0], points_rot[:, 1]

        x_ra, y_dec = (x_ra, y_dec) / source_params["R_sersic"]

        dists = (x_ra**2 + y_dec**2) ** 0.5 * source_params["R_sersic"]
        res = find_closest_point_indices(
            np.array([self.x_bins, self.y_bins]).T,
            np.array([x_ra, y_dec]).T,
            self.num_rsersics,
        )

        if return_dist:
            return res, dists
        return res

    def generate_residuals(self, all_fitted_params, return_datacube=False, linear_solve=False):
        assert self.get_num_free_params(linear_solve=linear_solve) == len(all_fitted_params)

        lens_model_params = all_fitted_params[: self.len_model_numparams]
        v_los_params = all_fitted_params[
            self.len_model_numparams : self.len_model_numparams + self.v_los_numparams
        ]
        
        # If linear_solve is True, v_disp is the end of the array. 
        # flx_params are omitted because we will solve for them analytically.
        if linear_solve:
            v_disp_params = all_fitted_params[
                self.len_model_numparams + self.v_los_numparams :
            ]
        else:
            v_disp_params = all_fitted_params[
                -1 * (self.flx_numparams + self.v_disp_numparams) : -1 * self.flx_numparams
            ]
            flx_params = all_fitted_params[-1 * self.flx_numparams :]

        kwargs_lenstronomy = self.init_fitting_seq.param_class.args2kwargs(
            lens_model_params
        )
        kwargs_lenstronomy.pop("kwargs_tracer_source", None)

        if np.any(
            (np.array(self.init_lenstronomy_args) - np.array(lens_model_params)) ** 2
            > 1e-8
        ):
            immodel = copy.deepcopy(self.imModel_classcreator)
            immodel.image_linear_solve(inv_bool=True, **kwargs_lenstronomy)
            immodel = immodel._imageModel_list[0]

            sm = immodel.source_mapping
            ra_grid, dec_grid = immodel.ImageNumerics.coordinates_evaluate
            x_source_vals, y_source_vals = sm._lens_model.ray_shooting(
                ra_grid, dec_grid, kwargs_lenstronomy["kwargs_lens"]
            )

        else:
            immodel = self.immodel_init._imageModel_list[0]
            x_source_vals, y_source_vals = (
                self.init_x_source_vals,
                self.init_y_source_vals,
            )
            sm = self.sm_init

        if self.num_bins > 0:
            binno = self.given_ra_dec_return_bin_no(
                x_source_vals, y_source_vals, kwargs_lenstronomy["kwargs_source"][0]
            )
        else:
            binno = np.ones(x_source_vals.shape)
            
        aux_params = [kwargs_lenstronomy["kwargs_source"], sm, self.constant_val, self.d_s]

        c = 299792
        z_los = (
            self.v_los_fnc(
                x_source_vals, y_source_vals, binno, aux_params, v_los_params
            )
            / c
        )
        v_disp = self.v_disp_fnc(
            x_source_vals, y_source_vals, binno, aux_params, v_disp_params
        )

        sigma_model = v_disp * self.central_wave / c
        sigma_total = (
            sigma_model**2 + (self.central_wave / (2.355 * self.spectral_res)) ** 2
        ) ** 0.5

        # ==========================================
        # LINEAR INVERSION BLOCK
        # ==========================================
        if linear_solve:
            unit_source_light = np.array(
                [
                    np.zeros(self.imset.wavelength.shape)
                    if np.sum(np.isnan([z, sigma_ang, 1.0])) > 0
                    else self.imset.gen_2d_spec_fixratios([z, sigma_ang, 1.0])
                    for z, sigma_ang in zip(z_los, sigma_total)
                ]
            )

            mask_bool = self.datacube_mask.astype(bool)
            valid_pixels = mask_bool # Keeping your custom masking rule
            
            # Count pixels to pre-allocate exact matrix size
            num_valid_pixels = np.sum(valid_pixels)
            
            # Cast W and b_data down to 32-bit floats
            W = (1.0 / np.sqrt(self.datacube_unc)).astype(np.float32)
            b_data = (self.obs_datacube[valid_pixels] * W).astype(np.float32)

            # Pre-allocate A_matrix as a 32-bit float array
            A_matrix = np.empty((num_valid_pixels, self.flx_numparams), dtype=np.float32)
            
            for k in range(self.flx_numparams):
                test_flx = np.zeros(self.flx_numparams)
                test_flx[k] = 1.0 
                basis_flxs = self.flx_fnc(x_source_vals, y_source_vals, binno, aux_params, test_flx)
                
                basis_source_light = unit_source_light * basis_flxs[:, np.newaxis]
                
                basis_datacube = np.zeros_like(self.obs_datacube)
                for ii in range(basis_source_light.shape[1]):
                    basis_datacube[ii] = immodel.ImageNumerics.re_size_convolve(
                        basis_source_light[:, ii], unconvolved=False
                    )
                
                # Assign directly to pre-allocated matrix and ensure it's a 32-bit float
                A_matrix[:, k] = (basis_datacube[valid_pixels] * W).astype(np.float32)
                
                # Aggressive memory cleanup
                del basis_source_light
                del basis_datacube
                if k % 10 == 0:
                    gc.collect()
            
            flx_params, _ = sp.optimize.nnls(A_matrix, b_data)
            
            # Clean up the large matrix right after solving
            del A_matrix
            del W
            gc.collect()

        # ==========================================
        # STANDARD MODEL GENERATION
        # ==========================================
        flxs = self.flx_fnc(x_source_vals, y_source_vals, binno, aux_params, flx_params)

        source_light = np.array(
            [
                np.zeros(self.imset.wavelength.shape)
                if np.sum(np.isnan([z, sigma_ang, flx])) > 0
                else self.imset.gen_2d_spec_fixratios([z, sigma_ang, flx])
                for z, sigma_ang, flx in zip(z_los, sigma_total, flxs)
            ]
        )

        model_datacube = []
        for ii in np.arange(source_light.shape[1]):
            model_datacube += [
                immodel.ImageNumerics.re_size_convolve(
                    source_light[:, ii], unconvolved=False
                )
            ]
        model_datacube = np.array(model_datacube)

        res = np.nansum(
            ((model_datacube - self.obs_datacube) ** 2 / self.datacube_unc) 
            * self.datacube_mask
        )

        if return_datacube:
            if linear_solve:
                return res, model_datacube, flx_params
            return res, model_datacube
        return res

    def generate_source_plots(self, all_fitted_params, image_size=None, dpix=None):
        assert self.get_num_free_params() == len(all_fitted_params)

        if image_size is None:
            image_size = self.sourceplane_size
        if dpix is None:
            dpix = self.dpix

        lens_model_params = all_fitted_params[: self.len_model_numparams]
        v_los_params = all_fitted_params[
            self.len_model_numparams : self.len_model_numparams + self.v_los_numparams
        ]
        v_disp_params = all_fitted_params[
            -1 * (self.flx_numparams + self.v_disp_numparams) : -1 * self.flx_numparams
        ]
        flx_params = all_fitted_params[-1 * self.flx_numparams :]

        kwargs_lenstronomy = self.init_fitting_seq.param_class.args2kwargs(
            lens_model_params
        )
        kwargs_lenstronomy.pop("kwargs_tracer_source", None)

        if np.any(
            (np.array(self.init_lenstronomy_args) - np.array(lens_model_params)) ** 2
            > 1e-8
        ):
            immodel = copy.deepcopy(self.imModel_classcreator)
            immodel.image_linear_solve(inv_bool=True, **kwargs_lenstronomy)
            immodel = immodel._imageModel_list[0]
            sm = immodel.source_mapping
        else:
            immodel = self.immodel_init._imageModel_list[0]
            sm = self.sm_init

        c = 299792
        delta_coor = (np.arange(image_size) - image_size / 2) * dpix

        v_los_img = np.zeros((image_size, image_size))
        v_disp_img = np.zeros((image_size, image_size))
        flxs_img = np.zeros((image_size, image_size))
        for ix, x in enumerate(
            kwargs_lenstronomy["kwargs_source"][0]["center_x"] + delta_coor
        ):
            for iy, y in enumerate(
                kwargs_lenstronomy["kwargs_source"][0]["center_y"] + delta_coor
            ):
                if self.num_bins > 0:
                    binno = self.given_ra_dec_return_bin_no(
                        np.array([x]),
                        np.array([y]),
                        kwargs_lenstronomy["kwargs_source"][0],
                    )
                else:
                    binno = 1
                aux_params = [
                    kwargs_lenstronomy["kwargs_source"],
                    sm,
                    self.constant_val,
                ]

                v_los = self.v_los_fnc(
                    np.array([x]), np.array([y]), binno, aux_params, v_los_params
                )[0]
                v_disp = self.v_disp_fnc(
                    np.array([x]), np.array([y]), binno, aux_params, v_disp_params
                )[0]
                flxs = self.flx_fnc(
                    np.array([x]), np.array([y]), binno, aux_params, flx_params
                )[0]

                if np.sum(np.isnan([v_los, v_disp, flxs])) > 0 or np.isnan(binno):
                    v_los_img[ix, iy] = np.nan
                    v_disp_img[ix, iy] = np.nan
                    flxs_img[ix, iy] = np.nan
                else:
                    v_los_img[ix, iy] = v_los
                    v_disp_img[ix, iy] = v_disp
                    flxs_img[ix, iy] = flxs
        v_los_img -= np.nanmedian(v_los_img)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        col = axs[0].imshow(
            v_los_img,
            cmap="bwr",
        )
        axs[0].invert_yaxis()
        fig.colorbar(col, ax=axs[0], label="LOS Velocity [km/s]")

        cmap = cm.get_cmap('viridis').copy()
        cmap.set_bad(color='black')

        col = axs[1].imshow(v_disp_img, cmap=cmap)
        axs[1].invert_yaxis()
        fig.colorbar(col, ax=axs[1], label="Velocity dispersion [km/s]")

        col = axs[2].imshow(np.log10(flxs_img), cmap=cmap)
        axs[2].invert_yaxis()
        fig.colorbar(col, ax=axs[2], label="log10 flux")
        plt.show()

    def decide_profiles_fnc(self, profile_name, num_bins):
        if profile_name == "VORONOI":
            return self.get_voronoi_v_given_xy_bin, num_bins
        elif profile_name == "ARCTAN":
            return self.get_arctan_v_given_xy_bin, 4
        elif profile_name == "SERSIC":
            return self.get_sersic_v_given_xy_bin, 1
        elif profile_name == "GAUSSIAN":
            return self.get_gaussian_v_given_xy_bin, 2
        elif profile_name == "POWER_LAW":
            return self.get_power_law_v_given_xy_bin, 2
        elif profile_name == "POWER_LAW_BH":
            return self.get_power_law_bh_v_given_xy_bin, 3
        elif profile_name == "CONSTANT_FIXED":
            return self.get_constant_v_given_xy_bin, 0
        elif profile_name == "CONSTANT_FITTED":
            return self.get_constant_v_given_xy_bin, 1
        raise Exception("Profile not implemented")

    @staticmethod
    def get_voronoi_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: list same len of num of bins
        if not check_list(x):
            if np.isnan(binno):
                return 0.0
            return fitted_params[int(binno)]
        else:
            return np.array(
                [0.0 if np.isnan(b) else fitted_params[int(b)] for b in binno]
            )

    @staticmethod
    def get_arctan_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [v_pa, v_a, v_b, v_c]
        kwargs_source = aux_params[0]
        v_pa, v_a, v_b, v_c = fitted_params
        c_x, c_y = kwargs_source[0]["center_x"], kwargs_source[0]["center_y"]

        if not check_list(x):
            return arctan_2d(v_pa, v_a, v_b, v_c, c_x, c_y, x, y)
        else:
            return np.array(
                [
                    arctan_2d(v_pa, v_a, v_b, v_c, c_x, c_y, xp, yp)
                    for xp, yp in zip(x, y)
                ]
            )

    @staticmethod
    def get_sersic_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [scale]
        kwargs_source, sm = aux_params[0], aux_params[1]
        scale = fitted_params[0]

        if not check_list(x):
            return (
                sm._light_model.surface_brightness(
                    np.array([x]), np.array([y]), kwargs_source
                )[0]
                * scale
            )
        else:
            return sm._light_model.surface_brightness(x, y, kwargs_source) * scale

    @staticmethod
    def get_gaussian_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [amp, sigma_model]
        if not check_list(x):
            kwargs_source = aux_params[0]
            amp, sigma_model = fitted_params

            c_x, c_y = kwargs_source[0]["center_x"], kwargs_source[0]["center_y"]
            dist = ((x - c_x) ** 2 + (y - c_y) ** 2) ** 0.5

            return norm_dist(amp, 0, sigma_intrinsic, dist)
        else:
            return np.array(
                [
                    get_gaussian_v_given_xy_bin(
                        x0, y0, binno0, aux_params, fitted_params
                    )
                    for x0, y0, binno0 in zip(x, y, binno)
                ]
            )

    @staticmethod
    def get_power_law_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [scale, gamma]
        if not check_list(x):
            return get_power_law_v_given_xy_bin(
                np.array([x]), np.array([y]), binno, aux_params, fitted_params
            )[0]
        else:
            kwargs_source = aux_params[0]
            scale, gamma = fitted_params

            x_, y_ = param_util.transform_e1e2_product_average(
                x - kwargs_source[0]["center_x"],
                y - kwargs_source[0]["center_y"],
                kwargs_source[0]["e1"],
                kwargs_source[0]["e2"],
                center_x=0,
                center_y=0,
            )
            dist = (x_**2 + y_**2) ** 0.5
            return scale * dist ** ((2 - gamma) / 2)

    @staticmethod
    def get_power_law_bh_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [scale, gamma, lg_bh_mass]
        if not check_list(x):
            return get_power_law_bh_v_given_xy_bin(
                np.array([x]), np.array([y]), binno, aux_params, fitted_params
            )[0]
        else:
            kwargs_source = aux_params[0]
            scale, gamma, lg_bh_mass = fitted_params

            x_, y_ = param_util.transform_e1e2_product_average(
                x - kwargs_source[0]["center_x"],
                y - kwargs_source[0]["center_y"],
                kwargs_source[0]["e1"],
                kwargs_source[0]["e2"],
                center_x=0,
                center_y=0,
            )
            dist = (x_**2 + y_**2) ** 0.5
            vd_power = scale * dist ** ((2 - gamma) / 2)

            G = 4.30241e-6 # in units of (km/s)^2 kpc/M_sol
            d_s = aux_params[3]
            dist = ((x - kwargs_source[0]["center_x"])**2 + (y - kwargs_source[0]["center_y"])**2) ** 0.5
            vd_bh_srd = (G*(10**lg_bh_mass)/(dist/206265*d_s))

            return (vd_power**2 + vd_bh_srd)**0.5
    
    @staticmethod
    def get_constant_v_given_xy_bin(x, y, binno, aux_params, fitted_params=[]):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [constant_val] or []
        if len(fitted_params) == 0:
            const_val = aux_params[2]
        else:
            const_val = fitted_params[0]

        if not check_list(x):
            return const_val
        else:
            return np.ones(len(x)) * const_val
