import os
import copy
import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

from .image_set import ImageSet
from .flat_modeling import FlatModel
from .iful_modeling import IFULModel
from .util import *

class SimulationMockImageSet(ImageSet):
    """
    Mock ImageSet configured for JWST NIRSpec G235M simulation.
    """
    def __init__(self, size, pixscale_arcsec, zs, wavelengths_full, psf_path):
        self.zs = zs
        self.size = size
        self.pixscale = pixscale_arcsec / 3600.0  # Pixel scale in degrees/pixel
        
        self.wavelengths_full = wavelengths_full
        N_wave = len(wavelengths_full)
        self.wavelength = self.wavelengths_full[5:-5]
        datacube = np.zeros((size, size, N_wave))
        
        self.continuum_subtraction(datacube, self.wavelengths_full, 0, 5)
        
        # Overwrite datacubes with positive values to avoid all-zero fits
        self.datacube = np.ones((size, size, N_wave - 10)) * 10.0
        self.datacube_whitelight = np.ones((size, size)) * 10.0
        
        self.brms_3d = 0.01
        self.brms_2d = 0.01
        
        # Mock 3D WCS Header for the datacube (RA, DEC, wavelength)
        header_wcs = Header()
        header_wcs["CTYPE1"] = "RA---TAN"
        header_wcs["CTYPE2"] = "DEC--TAN"
        header_wcs["CTYPE3"] = "AWAV"
        header_wcs["CRPIX1"] = (size + 1) / 2.0
        header_wcs["CRVAL1"] = 0.0
        header_wcs["CDELT1"] = -self.pixscale
        header_wcs["CUNIT1"] = "deg"
        header_wcs["CRPIX2"] = (size + 1) / 2.0
        header_wcs["CRVAL2"] = 0.0
        header_wcs["CDELT2"] = self.pixscale
        header_wcs["CUNIT2"] = "deg"
        header_wcs["CRPIX3"] = 1.0
        header_wcs["CRVAL3"] = self.wavelengths_full[5]
        header_wcs["CDELT3"] = self.wavelengths_full[1] - self.wavelengths_full[0]
        header_wcs["CUNIT3"] = "Angstrom"
        
        self.aux_info = {
            "header_wcs": header_wcs,
            "init_lens_center": np.array([[(size - 1) / 2.0, (size - 1) / 2.0]]),
            "pixscale": self.pixscale,
            "exptime": 1000.0,
            "final_psf": np.load(psf_path)
        }
        self.img_locations = np.array([])


def create_simulation_models(imset, theta_E=0.8, source_x=0.09, source_y=0.09, iful_profiles=None):
    """
    Initializes and configures the FlatModel and IFULModel instances for the simulation.
    """
    # Setup FlatModel
    fm = FlatModel(imset, ["EPL_Q_PHI", "SHEAR"], ["SERSIC_ELLIPSE"])
    
    kwargs_model = {
        "lens_model_list": ["EPL_Q_PHI", "SHEAR"],
        "source_light_model_list": ["SERSIC_ELLIPSE"],
        "lens_light_model_list": [],
        "point_source_model_list": [],
        "fixed_magnification_list": [],
    }

    kwargs_data_joint = {
        "multi_band_list": [fm.multi_band_list],
        "multi_band_type": "single-band",
    }

    kwargs_lens_init = [
        {"theta_E": theta_E, "gamma": 2.0, "q": 0.75, "phi": 0.5, "center_x": 0.0, "center_y": 0.0},
        {"gamma1": 0.05, "gamma2": -0.05, "ra_0": 0.0, "dec_0": 0.0}
    ]
    kwargs_lens_sigma = [
        {"theta_E": 0.05, "gamma": 0.05, "q": 0.02, "phi": 0.05, "center_x": 0.05, "center_y": 0.05},
        {"gamma1": 0.005, "gamma2": 0.005, "ra_0": 0.05, "dec_0": 0.05}
    ]
    fixed_lens = [{}, {"ra_0": 0.0, "dec_0": 0.0}]
    kwargs_lower_lens = [
        {"theta_E": 0.2, "gamma": 1.5, "q": 0.3, "phi": -np.pi, "center_x": -1.0, "center_y": -1.0},
        {"gamma1": -0.2, "gamma2": -0.2}
    ]
    kwargs_upper_lens = [
        {"theta_E": 1.5, "gamma": 2.5, "q": 1.0, "phi": np.pi, "center_x": 1.0, "center_y": 1.0},
        {"gamma1": 0.2, "gamma2": 0.2}
    ]

    kwargs_source_init = [{"amp": 1000.0, "R_sersic": 0.075, "n_sersic": 1.5, "e1": 0.0, "e2": 0.0, "center_x": source_x, "center_y": source_y}]
    kwargs_source_sigma = [{"amp": 100.0, "R_sersic": 0.02, "n_sersic": 0.1, "e1": 0.05, "e2": 0.05, "center_x": 0.05, "center_y": 0.05}]
    fixed_source = [{}]
    kwargs_lower_source = [{"amp": 0.0, "R_sersic": 0.005, "n_sersic": 0.5, "e1": -0.5, "e2": -0.5, "center_x": -1.0, "center_y": -1.0}]
    kwargs_upper_source = [{"amp": 1e6, "R_sersic": 1.0, "n_sersic": 8.0, "e1": 0.5, "e2": 0.5, "center_x": 1.0, "center_y": 1.0}]

    kwargs_params = {
        "lens_model": [kwargs_lens_init, kwargs_lens_sigma, fixed_lens, kwargs_lower_lens, kwargs_upper_lens],
        "source_model": [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source],
        "lens_light_model": [[], [], [], [], []],
        "point_source_model": [[], [], [], [], []],
    }

    fitting_seq = FittingSequence(
        kwargs_data_joint,
        kwargs_model,
        {},
        {},
        kwargs_params,
    )

    fm.init_fitting_seq = fitting_seq
    fm.init_pso_fit = {
        "kwargs_lens": kwargs_lens_init,
        "kwargs_source": kwargs_source_init,
    }

    # Setup IFULModel
    if iful_profiles is None:
        iful_profiles = ["ARCTAN", "CONSTANT_FITTED_BH", "SERSIC"]
    d_s = FlatLambdaCDM(H0=70, Om0=0.3).angular_diameter_distance(imset.zs).to(u.kpc).value

    ifulmodel = IFULModel(
        imset, fm, iful_profiles,
        sourceplane_size=100, num_bins=0, num_rsersics=1,
        spectral_res=1000, equal_weight_voronoi=False, d_s=d_s
    )

    return fm, ifulmodel


def run_galaxy_simulation(ifulmodel, sim_params, compute_window=3, grid_scale=0.01, source_grid_size=100, source_grid_scale=0.015):
    """
    Evaluates the lensing configuration and returns the lensed image, unlensed source,
    and caustic/critical curve coordinates.
    """
    lens_model_params = sim_params[: ifulmodel.len_model_numparams]
    kwargs_lenstronomy = ifulmodel.init_fitting_seq.param_class.args2kwargs(lens_model_params)
    immodel = (ifulmodel.immodel_init._imageModel_list if hasattr(ifulmodel.immodel_init, "_imageModel_list") else ifulmodel.immodel_init._image_model_list)[0]

    # Calculate critical curves and caustics
    lens_model = LensModel(lens_model_list=["EPL_Q_PHI", "SHEAR"])
    lens_model_ext = LensModelExtensions(lens_model)
    ra_crit, dec_crit, ra_caustic, dec_caustic = lens_model_ext.critical_curve_caustics(
        kwargs_lenstronomy["kwargs_lens"], compute_window=compute_window, grid_scale=grid_scale
    )

    # Evaluate unlensed source plane surface brightness
    x_grid_source, y_grid_source = np.meshgrid(
        (np.arange(source_grid_size) - source_grid_size / 2.0) * source_grid_scale,
        (np.arange(source_grid_size) - source_grid_size / 2.0) * source_grid_scale
    )
    unlensed_source = ifulmodel.sm_init._light_model.surface_brightness(
        x_grid_source, y_grid_source, kwargs_lenstronomy["kwargs_source"]
    )

    # Evaluate lensed image
    lensed_image = immodel.image(
        kwargs_lenstronomy["kwargs_lens"],
        kwargs_lenstronomy["kwargs_source"]
    )

    return lensed_image, unlensed_source, ra_crit, dec_crit, ra_caustic, dec_caustic


def add_instrument_noise(datacube, bg_noise_std_frac=0.02, seed=42):
    """
    Adds Gaussian background/readout noise to the datacube.
    """
    np.random.seed(seed)
    peak_flux = np.max(datacube)
    bg_std = bg_noise_std_frac * peak_flux
    bg_noise = np.random.normal(loc=0.0, scale=bg_std, size=datacube.shape)
    noisy_datacube = datacube + bg_noise
    return noisy_datacube, bg_noise


def export_to_fits(filename, datacube_noisy, wavelengths_full, header_wcs, redshift=3.8, exptime=1000.0, comment=None):
    """
    Saves the simulated datacube to a FITS file with proper WCS headers.
    """
    hdu = fits.PrimaryHDU(datacube_noisy)
    
    # Copy WCS headers
    for key in header_wcs.keys():
        if key not in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'EXTEND']:
            hdu.header[key] = header_wcs[key]
            
    hdu.header["EXPTIME"] = exptime
    hdu.header["REDSHIFT"] = redshift
    if comment is None:
        comment = "Simulated noisy lensed galaxy datacube"
    hdu.header["COMMENT"] = comment
    
    # Save to FITS
    hdu.writeto(filename, overwrite=True)