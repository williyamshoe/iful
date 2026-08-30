import pytest
import os
import numpy as np
from astropy.io import fits

from iful.simulation_api import (
    SimulationMockImageSet,
    create_simulation_models,
    add_instrument_noise,
    export_to_fits,
)

@pytest.fixture
def temp_psf_path(tmp_path):
    psf = np.ones((5, 5)) / 25.0
    psf_file = tmp_path / "test_psf.npy"
    np.save(psf_file, psf)
    return str(psf_file)

def test_simulation_mock_image_set(temp_psf_path):
    size = 12
    pixscale_arcsec = 0.05
    zs = 3.0
    waves_full = np.linspace(5000, 5200, 30)

    sim_imset = SimulationMockImageSet(
        size=size,
        pixscale_arcsec=pixscale_arcsec,
        zs=zs,
        wavelengths_full=waves_full,
        psf_path=temp_psf_path,
    )

    assert sim_imset.size == size
    assert sim_imset.zs == zs
    assert sim_imset.datacube.shape == (size, size, 20)
    assert "header_wcs" in sim_imset.aux_info
    assert "final_psf" in sim_imset.aux_info

def test_create_simulation_models(temp_psf_path):
    size = 10
    pixscale_arcsec = 0.05
    zs = 3.0
    waves_full = np.linspace(5000, 5200, 30)

    sim_imset = SimulationMockImageSet(
        size=size,
        pixscale_arcsec=pixscale_arcsec,
        zs=zs,
        wavelengths_full=waves_full,
        psf_path=temp_psf_path,
    )

    profiles = ["ARCTAN", "CONSTANT_FITTED_BH", "SERSIC"]
    fm, ifulmodel = create_simulation_models(
        sim_imset,
        theta_E=0.8,
        source_x=0.05,
        source_y=0.05,
        iful_profiles=profiles
    )

    assert fm is not None
    assert ifulmodel is not None
    assert ifulmodel.iful_profiles == profiles

def test_add_instrument_noise():
    cube = np.ones((10, 10, 5)) * 100.0
    noisy_cube, noise = add_instrument_noise(cube, bg_noise_std_frac=0.05, seed=42)
    assert noisy_cube.shape == cube.shape
    assert noise.shape == cube.shape
    assert not np.array_equal(noisy_cube, cube)

def test_export_to_fits(tmp_path, temp_psf_path):
    size = 10
    waves_full = np.linspace(5000, 5200, 30)
    sim_imset = SimulationMockImageSet(
        size=size,
        pixscale_arcsec=0.05,
        zs=3.0,
        wavelengths_full=waves_full,
        psf_path=temp_psf_path,
    )

    fits_file = tmp_path / "test_out.fits"
    export_to_fits(
        filename=str(fits_file),
        datacube_noisy=sim_imset.datacube,
        wavelengths_full=sim_imset.wavelength,
        header_wcs=sim_imset.aux_info["header_wcs"],
        redshift=3.0,
        exptime=1000.0
    )

    assert os.path.exists(fits_file)
    with fits.open(fits_file) as hdul:
        assert hdul[0].data.shape == sim_imset.datacube.shape
        assert hdul[0].header["REDSHIFT"] == 3.0
