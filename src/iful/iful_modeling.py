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
from powerbin import PowerBin
from tqdm import tqdm
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
from lenstronomy.Util import param_util
import math

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
            self.d_s = FlatLambdaCDM(H0=70, Om0=0.3).angular_diameter_distance(imageset.zs).to(u.kpc).value
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
        self.sm_init = (self.immodel_init._imageModel_list if hasattr(self.immodel_init, "_imageModel_list") else self.immodel_init._image_model_list)[0].source_mapping

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
            source_fluxes_arg = np.log10(copy.deepcopy(self.source_fluxes))
            self.voronoi_given_nbins(
                num_bins,
                np.nanmax(source_fluxes_arg) * 2,
                np.nansum(source_fluxes_arg) / np.sum(~np.isnan(source_fluxes_arg)) ** 0.5 / 2,
                flatmodel.init_pso_fit["kwargs_source"],
                source_fluxes_arg
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

        self.datacube_mask = np.transpose(self.imset.mask_3d, (2, 0, 1))
        
        exptime = self.imset.aux_info.get("exptime", 1.0)
        poisson_obs = np.maximum(self.obs_datacube, 0.0) / exptime
        datacube_variance = (self.imset.brms_3d)**2 + (poisson_obs)
        datacube_variance = np.where(
            np.isnan(datacube_variance) | (datacube_variance <= 0),
            1e10,
            datacube_variance
        )
        self.datacube_unc = np.sqrt(datacube_variance)

        self.central_wave = np.mean(self.imset.wavelength)

        self.init_lenstronomy_args = self.init_fitting_seq.param_class.kwargs2args(
            **flatmodel.init_pso_fit
        )

        ra_grid, dec_grid = (self.immodel_init._imageModel_list if hasattr(self.immodel_init, "_imageModel_list") else self.immodel_init._image_model_list)[
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
        elif self.iful_profiles[-1].startswith("SHAPELETS"):
            num_params += 1
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
        xy = np.column_stack((self.pixel_locations.T[1], self.pixel_locations.T[0]))

        def capacity_spec(index):
            sn = np.sum(source_fluxes_arg[index]) / np.sqrt(len(index))
            return sn**2

        while (high - low) > epsilon:
            mid = low + (high - low) / 2.0
            pow_bin = PowerBin(
                xy,
                capacity_spec,
                target_capacity=mid**2,
                pixelsize=None,
                verbose=0,
                regul=True,
            )
            mid_y = len(pow_bin.xybin)
            if mid_y == target_y:
                break
            elif mid_y < target_y:
                high = mid
            else:
                low = mid

        pow_bin = PowerBin(
            xy,
            capacity_spec,
            target_capacity=mid**2,
            pixelsize=None,
            verbose=0,
            regul=True,
        )
        pow_bin.plot(capacity_scale='sqrt')

        bin_number = pow_bin.bin_num
        y_gen = pow_bin.xybin[:, 0]
        x_gen = pow_bin.xybin[:, 1]
        y_bar = pow_bin.xybin[:, 0]
        x_bar = pow_bin.xybin[:, 1]
        sn = np.sqrt(pow_bin.bin_capacity)
        nPixels = pow_bin.npix
        scale = None

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

    def generate_residuals(self, all_fitted_params, return_datacube=False, linear_solve=False, vd_plots=False, trim_vd_plot=0):
        assert self.get_num_free_params(linear_solve=linear_solve) == len(all_fitted_params)

        lens_model_params = all_fitted_params[: self.len_model_numparams]
        v_los_params = all_fitted_params[
            self.len_model_numparams : self.len_model_numparams + self.v_los_numparams
        ]
        
        # If linear_solve is True, v_disp is the end of the array. 
        # flx_params are omitted because we will solve for them analytically.
        if linear_solve and not self.iful_profiles[-1].startswith("SHAPELETS"):
            v_disp_params = all_fitted_params[self.len_model_numparams + self.v_los_numparams :]
            flx_params_base = []
            num_linparam = self.flx_numparams
        elif linear_solve and self.iful_profiles[-1].startswith("SHAPELETS"):
            v_disp_params = all_fitted_params[self.len_model_numparams + self.v_los_numparams : -1]
            flx_params_base = list(all_fitted_params[-1:])
            num_linparam = self.flx_numparams - 1
        else:
            v_disp_params = all_fitted_params[-1 * (self.flx_numparams + self.v_disp_numparams) : -1 * self.flx_numparams]
            flx_params_base = list(all_fitted_params[-1 * self.flx_numparams :])
            num_linparam = 0
            flx_params = np.array(flx_params_base)

        kwargs_lenstronomy = self.init_fitting_seq.param_class.args2kwargs(lens_model_params)
        kwargs_lenstronomy.pop("kwargs_tracer_source", None)

        if np.any((np.array(self.init_lenstronomy_args) - np.array(lens_model_params)) ** 2 > 1e-8):
            immodel = copy.deepcopy(self.imModel_classcreator)
            immodel.image_linear_solve(inv_bool=True, **kwargs_lenstronomy)
            immodel = (immodel._imageModel_list if hasattr(immodel, "_imageModel_list") else immodel._image_model_list)[0]

            sm = immodel.source_mapping
            ra_grid, dec_grid = immodel.ImageNumerics.coordinates_evaluate
            x_source_vals, y_source_vals = sm._lens_model.ray_shooting(ra_grid, dec_grid, kwargs_lenstronomy["kwargs_lens"])

        else:
            immodel = (self.immodel_init._imageModel_list if hasattr(self.immodel_init, "_imageModel_list") else self.immodel_init._image_model_list)[0]
            x_source_vals, y_source_vals = (self.init_x_source_vals, self.init_y_source_vals,)
            sm = self.sm_init

        if self.num_bins > 0:
            binno = self.given_ra_dec_return_bin_no(x_source_vals, y_source_vals, kwargs_lenstronomy["kwargs_source"][0])
        else:
            binno = np.ones(x_source_vals.shape)
            
        aux_params = [kwargs_lenstronomy["kwargs_source"], sm, self.constant_val, self.d_s]

        c = 299792
        z_los = self.v_los_fnc(
            x_source_vals, y_source_vals, binno, aux_params, v_los_params
        ) / c
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
            # Vectorized unit_source_light calculation
            w = self.imset.wavelength
            rp = np.array(self.imset.restwave_peaks)
            ratios_f = np.array([1] + list(self.imset.init_spec_fit[3:]))

            z_grid = z_los[:, np.newaxis, np.newaxis]
            sig_grid = sigma_total[:, np.newaxis, np.newaxis]
            w_grid = w[np.newaxis, :, np.newaxis]
            rp_grid = rp[np.newaxis, np.newaxis, :]
            ratio_grid = ratios_f[np.newaxis, np.newaxis, :]

            mu = rp_grid * (1.0 + z_grid)
            diff = w_grid - mu
            sig_grid_safe = np.where(sig_grid <= 0, 1.0, sig_grid)
            diff_sig = diff / sig_grid_safe
            gauss = ratio_grid * np.exp(-0.5 * diff_sig**2)
            unit_source_light = np.sum(gauss, axis=2)

            nan_mask = np.isnan(z_los) | np.isnan(sigma_total) | (sigma_total <= 0)
            unit_source_light[nan_mask] = 0.0

            mask_bool = self.datacube_mask.astype(bool)
            valid_pixels = mask_bool # Keeping your custom masking rule
            
            # Count pixels to pre-allocate exact matrix size
            num_valid_pixels = np.sum(valid_pixels)
            
            # Cast W and b_data down to 32-bit floats
            if isinstance(self.datacube_unc, np.ndarray):
                W = (1.0 / self.datacube_unc).astype(np.float32)
                b_data = (self.obs_datacube[valid_pixels] * W[valid_pixels]).astype(np.float32)
            else:
                W = (1.0 / self.datacube_unc).astype(np.float32)
                b_data = (self.obs_datacube[valid_pixels] * W).astype(np.float32)

            # Pre-allocate A_matrix as a 32-bit float array
            A_matrix = np.empty((num_valid_pixels, num_linparam), dtype=np.float32)
            
            for k in range(num_linparam):
                test_flx = np.zeros(num_linparam)
                test_flx[k] = 1.0 
                test_flx = np.array(flx_params_base + list(test_flx))
                basis_flxs = self.flx_fnc(x_source_vals, y_source_vals, binno, aux_params, test_flx)
                
                basis_source_light = unit_source_light * basis_flxs[:, np.newaxis]
                
                basis_datacube = np.zeros_like(self.obs_datacube)
                for ii in range(basis_source_light.shape[1]):
                    basis_datacube[ii] = immodel.ImageNumerics.re_size_convolve(
                        basis_source_light[:, ii], unconvolved=False
                    )
                
                # Assign directly to pre-allocated matrix and ensure it's a 32-bit float
                if isinstance(self.datacube_unc, np.ndarray):
                    A_matrix[:, k] = (basis_datacube[valid_pixels] * W[valid_pixels]).astype(np.float32)
                else:
                    A_matrix[:, k] = (basis_datacube[valid_pixels] * W).astype(np.float32)
                
                # Aggressive memory cleanup
                del basis_source_light
                del basis_datacube
                if k % 10 == 0:
                    gc.collect()
            
            try:
                # Cholesky NNLS for extreme speedup on large matrices
                C = A_matrix.T @ A_matrix
                C.flat[::num_linparam + 1] += 1e-12
                v = A_matrix.T @ b_data
                R = sp.linalg.cholesky(C, lower=False)
                d = sp.linalg.solve_triangular(R, v, trans='T', lower=False)
                flx_params, _ = sp.optimize.nnls(R, d)
            except sp.linalg.LinAlgError:
                flx_params, _ = sp.optimize.nnls(A_matrix, b_data)
            flx_params = np.array(list(flx_params_base) + list(flx_params))
            
            # Clean up the large matrix right after solving
            del A_matrix
            del W
            gc.collect()

        # ==========================================
        # STANDARD MODEL GENERATION
        # ==========================================
        flxs = self.flx_fnc(x_source_vals, y_source_vals, binno, aux_params, flx_params)

        # Vectorized source_light calculation
        w = self.imset.wavelength
        rp = np.array(self.imset.restwave_peaks)
        ratios_f = np.array([1] + list(self.imset.init_spec_fit[3:]))

        z_grid = z_los[:, np.newaxis, np.newaxis]
        sig_grid = sigma_total[:, np.newaxis, np.newaxis]
        w_grid = w[np.newaxis, :, np.newaxis]
        rp_grid = rp[np.newaxis, np.newaxis, :]
        ratio_grid = ratios_f[np.newaxis, np.newaxis, :]
        flx_grid = flxs[:, np.newaxis, np.newaxis]

        mu = rp_grid * (1.0 + z_grid)
        diff = w_grid - mu
        sig_grid_safe = np.where(sig_grid <= 0, 1.0, sig_grid)
        diff_sig = diff / sig_grid_safe
        gauss = flx_grid * ratio_grid * np.exp(-0.5 * diff_sig**2)
        source_light = np.sum(gauss, axis=2)

        nan_mask = np.isnan(z_los) | np.isnan(sigma_total) | np.isnan(flxs) | (sigma_total <= 0)
        source_light[nan_mask] = 0.0

        model_datacube = []
        for ii in np.arange(source_light.shape[1]):
            model_datacube += [
                immodel.ImageNumerics.re_size_convolve(
                    source_light[:, ii], unconvolved=False
                )
            ]
        model_datacube = np.array(model_datacube)

        res = np.nansum(
            ((model_datacube - self.obs_datacube) ** 2 / self.datacube_unc**2) 
            * self.datacube_mask
        )

        if vd_plots:    
            lensed_diag_imgs = np.array(
                [
                    [1, z * c - v_los_params[-1], vds]
                    for z, vds in zip(z_los, v_disp)
                ]
            )
            diag_plots = []
            magn = np.mean(immodel.ImageNumerics.re_size_convolve(lensed_diag_imgs[:, 0], unconvolved=False))
            diag_plots += [
                immodel.ImageNumerics.re_size_convolve(
                    lensed_diag_imgs[:, 1]/magn, unconvolved=False
                )
            ]
            vd_flat = lensed_diag_imgs[:, 2]
            vd_flat = np.array([v if v<np.percentile(vd_flat, 99.99) else np.mean(vd_flat) for v in vd_flat])
            diag_plots += [
                immodel.ImageNumerics.re_size_convolve(
                    vd_flat**2/magn, unconvolved=False
                )**0.5
            ]
            # diag_plots += [
            #     immodel.ImageNumerics.re_size_convolve(
            #         lensed_diag_imgs[:, 2]/magn, unconvolved=False
            #     )
            # ]
            diag_plots = np.array(diag_plots)

            # binary_mask = np.where(np.sum(model_datacube, axis=0) > , 1.0, np.nan)

            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            
            col = axs[0].imshow(diag_plots[0, :, :], cmap="bwr")
            axs[0].set_axis_off()
            axs[0].invert_yaxis()
            fig.colorbar(col, ax=axs[0], label="LOS (convolved)")

            vd_plot = diag_plots[1, :, :]

            if trim_vd_plot > 0:
                vd_plot[:trim_vd_plot, :] = np.nan
                vd_plot[-trim_vd_plot:, :] = np.nan
                vd_plot[:, :trim_vd_plot] = np.nan
                vd_plot[:, -trim_vd_plot:] = np.nan
                
            col = axs[1].imshow(vd_plot)
            axs[1].set_axis_off()
            axs[1].invert_yaxis()
            fig.colorbar(col, ax=axs[1], label="velocity dispersion (convolved)")
            
            col = axs[2].imshow(np.sum(model_datacube, axis=0))
            axs[2].set_axis_off()
            axs[2].invert_yaxis()
            fig.colorbar(col, ax=axs[2], label="flux")
        
        if return_datacube:
            if linear_solve:
                return res, model_datacube, flx_params
            return res, model_datacube
        return res

    def generate_image_residuals(self, all_fitted_params, return_image=False, linear_solve=False):
        # The number of expected parameters is just the lens parameters + flux parameters
        expected_params = self.len_model_numparams
        if not linear_solve:
            expected_params += self.flx_numparams
        elif self.iful_profiles[-1].startswith("SHAPELETS"):
            expected_params += 1
            
        assert expected_params == len(all_fitted_params), "Mismatch in number of fitted parameters."

        lens_model_params = all_fitted_params[: self.len_model_numparams]
        
        # Parse flux parameters without v_los or v_disp
        if linear_solve and not self.iful_profiles[-1].startswith("SHAPELETS"):
            flx_params_base = []
            num_linparam = self.flx_numparams
        elif linear_solve and self.iful_profiles[-1].startswith("SHAPELETS"):
            flx_params_base = list(all_fitted_params[-1:])
            num_linparam = self.flx_numparams - 1
        else:
            flx_params_base = list(all_fitted_params[self.len_model_numparams :])
            num_linparam = 0
            flx_params = np.array(flx_params_base)

        kwargs_lenstronomy = self.init_fitting_seq.param_class.args2kwargs(lens_model_params)
        kwargs_lenstronomy.pop("kwargs_tracer_source", None)

        if np.any((np.array(self.init_lenstronomy_args) - np.array(lens_model_params)) ** 2 > 1e-8):
            immodel = copy.deepcopy(self.imModel_classcreator)
            immodel.image_linear_solve(inv_bool=True, **kwargs_lenstronomy)
            immodel = (immodel._imageModel_list if hasattr(immodel, "_imageModel_list") else immodel._image_model_list)[0]

            sm = immodel.source_mapping
            ra_grid, dec_grid = immodel.ImageNumerics.coordinates_evaluate
            x_source_vals, y_source_vals = sm._lens_model.ray_shooting(ra_grid, dec_grid, kwargs_lenstronomy["kwargs_lens"])

        else:
            immodel = (self.immodel_init._imageModel_list if hasattr(self.immodel_init, "_imageModel_list") else self.immodel_init._image_model_list)[0]
            x_source_vals, y_source_vals = (self.init_x_source_vals, self.init_y_source_vals,)
            sm = self.sm_init

        if self.num_bins > 0:
            binno = self.given_ra_dec_return_bin_no(x_source_vals, y_source_vals, kwargs_lenstronomy["kwargs_source"][0])
        else:
            binno = np.ones(x_source_vals.shape)
            
        aux_params = [kwargs_lenstronomy["kwargs_source"], sm, self.constant_val, self.d_s]

        obs_image = self.imset.datacube_whitelight
        unc_image = self.imset.brms_2d
        mask_bool = self.imset.mask.astype(bool)

        # ==========================================
        # LINEAR INVERSION BLOCK
        # ==========================================
        
        if linear_solve:
            valid_pixels = mask_bool 
            
            # Count pixels to pre-allocate exact matrix size
            num_valid_pixels = np.sum(valid_pixels)
            
            # Cast W and b_data down to 32-bit floats
            W = (1.0 / np.sqrt(unc_image)).astype(np.float32)
            b_data = (obs_image[valid_pixels] * W).astype(np.float32)

            # Pre-allocate A_matrix as a 32-bit float array
            A_matrix = np.empty((num_valid_pixels, num_linparam), dtype=np.float32)
            
            for k in range(num_linparam):
                test_flx = np.zeros(num_linparam)
                test_flx[k] = 1.0 
                test_flx = np.array(flx_params_base + list(test_flx))
                basis_flxs = self.flx_fnc(x_source_vals, y_source_vals, binno, aux_params, test_flx)
                
                # Directly resize and convolve the 2D basis model
                basis_image = immodel.ImageNumerics.re_size_convolve(
                    basis_flxs, unconvolved=False
                )
                
                # Assign directly to pre-allocated matrix and ensure it's a 32-bit float
                A_matrix[:, k] = (basis_image[valid_pixels] * W).astype(np.float32)
                
                # Aggressive memory cleanup
                del basis_flxs
                del basis_image
                if k % 10 == 0:
                    gc.collect()
            
            try:
                # Cholesky NNLS for extreme speedup on large matrices
                C = A_matrix.T @ A_matrix
                C.flat[::num_linparam + 1] += 1e-12
                v = A_matrix.T @ b_data
                R = sp.linalg.cholesky(C, lower=False)
                d = sp.linalg.solve_triangular(R, v, trans='T', lower=False)
                flx_params, _ = sp.optimize.nnls(R, d)
            except sp.linalg.LinAlgError:
                flx_params, _ = sp.optimize.nnls(A_matrix, b_data)
            flx_params = np.array(list(flx_params_base) + list(flx_params))
            
            # Clean up the large matrix right after solving
            del A_matrix
            del W
            gc.collect()

        # ==========================================
        # STANDARD MODEL GENERATION
        # ==========================================
        flxs = self.flx_fnc(x_source_vals, y_source_vals, binno, aux_params, flx_params)

        # Generate single 2D model image
        model_image = immodel.ImageNumerics.re_size_convolve(
            flxs, unconvolved=False
        )

        res = np.nansum(
            ((model_image - obs_image) ** 2 / unc_image) 
            * self.imset.mask
        )

        if return_image:
            if linear_solve:
                return res, model_image, flx_params
            return res, model_image
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

        if np.any((np.array(self.init_lenstronomy_args) - np.array(lens_model_params)) ** 2 > 1e-8):
            immodel = copy.deepcopy(self.imModel_classcreator)
            immodel.image_linear_solve(inv_bool=True, **kwargs_lenstronomy)
            immodel = (immodel._imageModel_list if hasattr(immodel, "_imageModel_list") else immodel._image_model_list)[0]
            sm = immodel.source_mapping
        else:
            immodel = (self.immodel_init._imageModel_list if hasattr(self.immodel_init, "_imageModel_list") else self.immodel_init._image_model_list)[0]
            sm = self.sm_init

        # c = 299792
        delta_coor = (np.arange(image_size) - image_size / 2) * dpix

        v_los_img = np.zeros((image_size, image_size))
        v_disp_img = np.zeros((image_size, image_size))
        flxs_img = np.zeros((image_size, image_size))
        aux_params = [kwargs_lenstronomy["kwargs_source"], sm, self.constant_val, self.d_s]
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

                v_los = self.v_los_fnc(np.array([x]), np.array([y]), binno, aux_params, v_los_params)[0]
                v_disp = self.v_disp_fnc(np.array([x]), np.array([y]), binno, aux_params, v_disp_params)[0]
                flxs = self.flx_fnc(np.array([x]), np.array([y]), binno, aux_params, flx_params)[0]

                if np.sum(np.isnan([v_los, v_disp, flxs])) > 0 or np.isnan(binno):
                    v_los_img[ix, iy] = np.nan
                    v_disp_img[ix, iy] = np.nan
                    flxs_img[ix, iy] = np.nan
                elif v_disp > 500:
                    v_los_img[ix, iy] = v_los
                    v_disp_img[ix, iy] = np.nan
                    flxs_img[ix, iy] = flxs
                else:
                    v_los_img[ix, iy] = v_los
                    v_disp_img[ix, iy] = v_disp
                    flxs_img[ix, iy] = flxs
        v_los_img -= np.nanmedian(v_los_img)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        col = axs[0].imshow(v_los_img, cmap="bwr")
        axs[0].invert_yaxis()
        fig.colorbar(col, ax=axs[0], label="LOS Velocity [km/s]")

        cmap = cm.get_cmap('viridis').copy()
        cmap.set_bad(color='black')

        # flat_idx = np.argsort(v_disp_img, axis=None)[-1:]
        # row_idx, col_idx = np.unravel_index(flat_idx, v_disp_img.shape)
        # v_disp_img[row_idx, col_idx] = np.nan

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
        elif profile_name == "EXPONENTIAL":
            return self.get_exponential_v_given_xy_bin, 2
        elif profile_name == "POWER_LAW_BH":
            return self.get_power_law_bh_v_given_xy_bin, 3
        elif profile_name == "CONSTANT_FIXED":
            return self.get_constant_v_given_xy_bin, 0
        elif profile_name == "CONSTANT_FITTED":
            return self.get_constant_v_given_xy_bin, 1
        elif profile_name == "CONSTANT_FITTED_BH":
            return self.get_constant_bh_v_given_xy_bin, 2
        elif profile_name == "TANH":
            return self.get_tanh_v_given_xy_bin, 4
        elif profile_name == "MULTIPARAM":
            return self.get_multiparam_v_given_xy_bin, 6
        elif profile_name.startswith("SHAPELETS"):
            try:
                n_max = int(profile_name.split("_")[1])
            except (IndexError, ValueError):
                n_max = 4  
            num_params = int((n_max + 1) * (n_max + 2) / 2)
            return self.get_shapelets_v_given_xy_bin, num_params + 1
            
        raise Exception("Profile not implemented")
        
    @staticmethod
    def get_voronoi_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: list same len of num of bins

        binno = np.asarray(binno)
        fitted_params = np.asarray(fitted_params)
        res = np.zeros_like(binno)
        nan_mask = np.isnan(binno)
        res[~nan_mask] = fitted_params[binno[~nan_mask].astype(int)]
        return res

    @staticmethod
    def get_arctan_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [v_pa, v_a, v_b, v_c]

        x = np.asarray(x)
        y = np.asarray(y)

        kwargs_source = aux_params[0]
        v_pa, v_a, v_b, v_c = fitted_params
        c_x, c_y = kwargs_source[0]["center_x"], kwargs_source[0]["center_y"]

        return arctan_2d(v_pa, v_a, v_b, v_c, c_x, c_y, x, y)

    @staticmethod
    def get_tanh_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [v_pa, v_a, v_b, v_c]

        x = np.asarray(x)
        y = np.asarray(y)

        kwargs_source = aux_params[0]
        v_pa, v_a, v_b, v_c = fitted_params
        c_x, c_y = kwargs_source[0]["center_x"], kwargs_source[0]["center_y"]

        return tanh_2d(v_pa, v_a, v_b, v_c, c_x, c_y, x, y)

    @staticmethod
    def get_multiparam_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [v_pa, v_a, v_b, v_beta, v_xi, v_c]

        x = np.asarray(x)
        y = np.asarray(y)

        kwargs_source = aux_params[0]
        v_pa, v_a, v_b, v_beta, v_xi, v_c = fitted_params
        c_x, c_y = kwargs_source[0]["center_x"], kwargs_source[0]["center_y"]

        return multiparam_2d(v_pa, v_a, v_b, v_beta, v_xi, v_c, c_x, c_y, x, y)

    @staticmethod
    def get_sersic_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [scale]

        x = np.array([x]) if not isinstance(x, (list, np.ndarray)) else np.array(x)
        y = np.array([y]) if not isinstance(y, (list, np.ndarray)) else np.array(y)

        kwargs_source, sm = aux_params[0], aux_params[1]
        scale = fitted_params[0]
        
        return sm._light_model.surface_brightness(x, y, kwargs_source) * scale

    @staticmethod
    def get_gaussian_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [amp, sigma_model]

        x = np.array([x]) if not isinstance(x, (list, np.ndarray)) else np.array(x)
        y = np.array([y]) if not isinstance(y, (list, np.ndarray)) else np.array(y)

        kwargs_source = aux_params[0]
        amp, sigma_model = fitted_params

        c_x, c_y = kwargs_source[0]["center_x"], kwargs_source[0]["center_y"]
        dist = ((x - c_x) ** 2 + (y - c_y) ** 2) ** 0.5

        return norm_dist(amp, 0, sigma_intrinsic, dist)
            
    @staticmethod
    def get_power_law_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [scale, gamma]

        x = np.array([x]) if not isinstance(x, (list, np.ndarray)) else np.array(x)
        y = np.array([y]) if not isinstance(y, (list, np.ndarray)) else np.array(y)

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
    def get_exponential_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [central_vd, scale_rad]

        x = np.array([x]) if not isinstance(x, (list, np.ndarray)) else np.array(x)
        y = np.array([y]) if not isinstance(y, (list, np.ndarray)) else np.array(y)

        kwargs_source = aux_params[0]
        central_vd, scale_rad = fitted_params

        x_, y_ = param_util.transform_e1e2_product_average(
            x - kwargs_source[0]["center_x"],
            y - kwargs_source[0]["center_y"],
            kwargs_source[0]["e1"],
            kwargs_source[0]["e2"],
            center_x=0,
            center_y=0,
        )
        dist = (x_**2 + y_**2) ** 0.5
        return central_vd * np.exp(-1 * dist / scale_rad)

    @staticmethod
    def get_power_law_bh_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [scale, gamma, lg_bh_mass]

        x = np.array([x]) if not isinstance(x, (list, np.ndarray)) else np.array(x)
        y = np.array([y]) if not isinstance(y, (list, np.ndarray)) else np.array(y)

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
        epsilon = 1e-5
        dist = (((x - kwargs_source[0]["center_x"])**2 + (y - kwargs_source[0]["center_y"])**2) + epsilon**2) ** 0.5
        vd_bh_srd = (G*(10**lg_bh_mass)/(dist/206265*d_s))

        return (vd_power**2 + vd_bh_srd)**0.5
    
    @staticmethod
    def get_constant_v_given_xy_bin(x, y, binno, aux_params, fitted_params=[]):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [constant_val] or []

        x = np.array([x]) if not isinstance(x, (list, np.ndarray)) else np.array(x)
        y = np.array([y]) if not isinstance(y, (list, np.ndarray)) else np.array(y)
        
        if len(fitted_params) == 0:
            const_val = aux_params[2]
        else:
            const_val = fitted_params[0]

        return np.ones(len(x)) * const_val

    @staticmethod
    def get_constant_bh_v_given_xy_bin(x, y, binno, aux_params, fitted_params=[]):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: [constant_val] or []

        x = np.array([x]) if not isinstance(x, (list, np.ndarray)) else np.array(x)
        y = np.array([y]) if not isinstance(y, (list, np.ndarray)) else np.array(y)

        kwargs_source = aux_params[0]

        if len(fitted_params) == 1:
            lg_bh_mass = fitted_params[0]
            const_val = aux_params[2]
        else:
            const_val, lg_bh_mass = fitted_params

        G = 4.30241e-6 # in units of (km/s)^2 kpc/M_sol
        d_s = aux_params[3]
        epsilon = 1e-5
        dist = (((x - kwargs_source[0]["center_x"])**2 + (y - kwargs_source[0]["center_y"])**2) + epsilon**2) ** 0.5
        vd_bh_srd = (G*(10**lg_bh_mass)/(dist/206265*d_s))

        vd_const = np.ones(len(x)) * const_val

        return (vd_const**2 + vd_bh_srd)**0.5

    @staticmethod
    def get_shapelets_v_given_xy_bin(x, y, binno, aux_params, fitted_params):
        # aux_params: [kwargs_source, sm, constant_val, d_s]
        # fitted_params: beta + 1D array of shapelet amplitudes
        kwargs_source = aux_params[0]
        beta = fitted_params[0]
        amp_array = fitted_params[1:]
        
        x = np.array([x]) if not isinstance(x, (list, np.ndarray)) else np.array(x)
        y = np.array([y]) if not isinstance(y, (list, np.ndarray)) else np.array(y)
        
        x_ell, y_ell = param_util.transform_e1e2_product_average(
            x - kwargs_source[0].get("center_x", 0.0),
            y - kwargs_source[0].get("center_y", 0.0),
            kwargs_source[0].get("e1", 0.0),
            kwargs_source[0].get("e2", 0.0),
            center_x=0,
            center_y=0,
        )
        
        n_max = int((-3 + math.sqrt(1 + 8 * len(amp_array))) / 2)
        
        shapelets = ShapeletSet()
        
        res = shapelets.function(x_ell, y_ell, amp_array, n_max, beta, 0.0, 0.0)
        
        return res
