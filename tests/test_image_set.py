import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

from iful.image_set import ImageSet

@pytest.fixture
def mock_datacube():
    size = 10
    n_wave = 30
    waves = np.linspace(5000, 5500, n_wave)
    # Create simple 3D datacube with continuum and emission line
    cube = np.ones((size, size, n_wave)) * 10.0
    # Add a mock emission line in the middle spaxels around channel 15
    cube[4:6, 4:6, 15] += 50.0
    return cube, waves

def test_image_set_init(mock_datacube):
    cube, waves = mock_datacube
    imset = ImageSet(
        datacube=cube,
        wavelengths=waves,
        zs=3.0,
        pixscale=0.05,
        gap=2,
        spectra_background=3
    )

    assert imset.zs == 3.0
    assert imset.pixscale == 0.05
    assert imset.size == 10
    assert imset.datacube.ndim == 3
    assert len(imset.wavelength) == len(waves) - 2 * (2 + 3)
    assert imset.datacube_whitelight.shape == (10, 10)
    assert imset.mask.shape == (10, 10)

def test_image_set_mask_and_aux(mock_datacube):
    cube, waves = mock_datacube
    imset = ImageSet(
        datacube=cube,
        wavelengths=waves,
        zs=3.0,
        pixscale=0.05,
        gap=1,
        spectra_background=2
    )

    # Test adding aux info
    imset.add_aux_info({"test_key": "test_val"})
    assert imset.aux_info["test_key"] == "test_val"

    # Test setting mask
    new_mask = np.zeros((10, 10))
    new_mask[3:7, 3:7] = 1.0
    imset.out_mask = np.ones((10, 10))
    imset.set_mask(new_mask)
    assert np.array_equal(imset.mask, new_mask)
    assert imset.mask_3d.shape == (10, 10, imset.datacube.shape[-1])

def test_image_set_spec_fit(mock_datacube):
    cube, waves = mock_datacube
    imset = ImageSet(
        datacube=cube,
        wavelengths=waves,
        zs=3.0,
        pixscale=0.05,
        gap=1,
        spectra_background=2
    )
    
    imset.mark_image_locations(np.array([[5, 5]]))
    assert len(imset.img_locations) == 1

    imset.get_initial_spec_fit_mask(img_size=2)
    assert len(imset.aperature_spec) == len(imset.wavelength)

    params = [3.0, 10.0, 100.0, 0.5] # z, sigma, amp, ratio
    imset.restwave_peaks = [1216.0]
    spec2d = imset.gen_2d_spec(params)
    assert len(spec2d) == len(imset.wavelength)
