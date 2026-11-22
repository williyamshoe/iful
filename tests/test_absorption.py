import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from iful.util import line_types_to_signs
from iful.image_set import ImageSet
from iful.simulation_api import SimulationMockImageSet, create_simulation_models, add_instrument_noise

C_KMS = 299792
ZS = 3.0
REST_WAVE = 1275.0  # lands at 5100 A, inside the mock bandpass


def test_line_types_to_signs():
    assert np.array_equal(line_types_to_signs(None, 2), [1.0, 1.0])
    assert np.array_equal(line_types_to_signs("absorption", 2), [-1.0, -1.0])
    assert np.array_equal(line_types_to_signs(["emission", "Absorption"], 2), [1.0, -1.0])
    with pytest.raises(ValueError):
        line_types_to_signs(["emission"], 2)
    with pytest.raises(ValueError):
        line_types_to_signs(["foo", "emission"], 2)


@pytest.fixture
def trough_cube():
    rng = np.random.default_rng(0)
    size, n_wave = 20, 30
    waves = np.linspace(5000, 5500, n_wave)
    cube = 10.0 + rng.normal(0, 0.1, (size, size, n_wave))
    cube[8:12, 8:12, 6:24] -= 5.0  # broad absorption trough in the central spaxels
    return cube, waves


def test_whitelight_sign(trough_cube):
    cube, waves = trough_cube
    emis = ImageSet(cube, waves, ZS, 0.05, 1, 2)
    absn = ImageSet(cube, waves, ZS, 0.05, 1, 2, line_types="absorption")

    assert emis.whitelight_sign == 1.0
    assert absn.whitelight_sign == -1.0
    assert emis.datacube_whitelight[10, 10] < 0
    assert absn.datacube_whitelight[10, 10] > 0
    np.testing.assert_allclose(absn.datacube_whitelight, -emis.datacube_whitelight)

    # set_line_types recomputes the white-light image
    emis.set_line_types("absorption")
    np.testing.assert_allclose(emis.datacube_whitelight, absn.datacube_whitelight)


def test_mask_outliers_keeps_trough(trough_cube):
    cube, waves = trough_cube
    emis = ImageSet(cube, waves, ZS, 0.05, 1, 2)
    absn = ImageSet(cube, waves, ZS, 0.05, 1, 2, line_types="absorption")
    emis.mask_outliers()
    absn.mask_outliers()
    # Treated as emission, the trough falls below the tight lower IQR bound and is masked
    assert emis.out_mask[10, 10] == 0.0
    assert absn.out_mask[10, 10] == 1.0


def test_gen_2d_spec_signs(trough_cube):
    cube, waves = trough_cube
    imset = ImageSet(cube, waves, ZS, 0.05, 1, 2, line_types=["absorption", "emission"])
    imset.restwave_peaks = [1270.0, 1290.0]
    spec = imset.gen_2d_spec([ZS, 20.0, 100.0, 1.0])
    assert spec[np.argmin(np.abs(imset.wavelength - 1270.0 * (1 + ZS)))] < -50
    assert spec[np.argmin(np.abs(imset.wavelength - 1290.0 * (1 + ZS)))] > 50


def _make_model(tmp_path, line_types, restwave_peaks=(REST_WAVE,), init_spec_fit=(ZS, 5.0, 1.0), n_wave=40):
    psf = np.ones((3, 3)) / 9.0
    psf_file = tmp_path / "psf.npy"
    np.save(psf_file, psf)
    imset = SimulationMockImageSet(
        size=30,
        pixscale_arcsec=0.05,
        zs=ZS,
        wavelengths_full=np.linspace(5000, 5200, n_wave),
        psf_path=str(psf_file),
        line_types=line_types,
    )
    imset.restwave_peaks = list(restwave_peaks)
    imset.init_spec_fit = np.array(init_spec_fit)
    _, ifulmodel = create_simulation_models(
        imset, theta_E=0.5, source_x=0.02, source_y=0.02,
        iful_profiles=["ARCTAN", "CONSTANT_FITTED_BH", "SERSIC"],
    )
    kin_params = [45.0, 100.0, 10.0, ZS * C_KMS, 60.0, 5.0]
    return ifulmodel, list(ifulmodel.init_lenstronomy_args) + kin_params


def test_absorption_model_is_negated_emission(tmp_path):
    emis_model, params = _make_model(tmp_path, None)
    absn_model, _ = _make_model(tmp_path, "absorption")

    _, cube_e = emis_model.generate_residuals(params + [1e3], return_datacube=True)
    _, cube_a = absn_model.generate_residuals(params + [1e3], return_datacube=True)

    assert cube_e.max() > 0
    assert cube_a.max() <= 0
    np.testing.assert_allclose(cube_a, -cube_e)


def test_absorption_linear_solve_recovers_depth(tmp_path):
    model, params = _make_model(tmp_path, "absorption")
    true_scale = 2.5e3
    _, cube = model.generate_residuals(params + [true_scale], return_datacube=True)

    model.obs_datacube = cube
    res, _, flx = model.generate_residuals(params, return_datacube=True, linear_solve=True)
    np.testing.assert_allclose(flx[0], true_scale, rtol=1e-3)
    assert res < 1e-6 * np.sum(cube**2 / model.datacube_unc**2)


def test_noise_on_absorption_cube():
    cube = -np.ones((5, 5, 5)) * 100.0
    noisy, noise = add_instrument_noise(cube, bg_noise_std_frac=0.05, seed=1)
    assert np.std(noise) > 1.0


# --- Scenario from the "Simulating an Absorption-Line Source" notebook section ---

TRIPLET = (1268.0, 1272.0, 1280.0)  # Mg b-like triplet shifted into the mock bandpass


def test_absorption_triplet_noisy_recovery(tmp_path):
    """Multi-line absorption cube + noise: the linear solve recovers the input depth."""
    model, params = _make_model(
        tmp_path, "absorption", restwave_peaks=TRIPLET, init_spec_fit=(ZS, 5.0, 1.0, 1.3, 1.5), n_wave=80
    )
    true_scale = 1e4
    _, cube = model.generate_residuals(params + [true_scale], return_datacube=True)
    assert cube.max() <= 0

    noisy, noise = add_instrument_noise(cube, bg_noise_std_frac=0.02, seed=42)
    assert np.std(noise) > 0

    model.obs_datacube = noisy
    _, _, flx = model.generate_residuals(params, return_datacube=True, linear_solve=True)
    np.testing.assert_allclose(flx[0], true_scale, rtol=0.01)


def test_mixed_line_types_model_signs(tmp_path):
    """Emission lines are added and absorption lines subtracted in the same model cube."""
    blue, red = 1262.0, 1288.0
    model, params = _make_model(
        tmp_path, ["emission", "absorption"], restwave_peaks=(blue, red),
        init_spec_fit=(ZS, 5.0, 1.0, 1.0), n_wave=80,
    )
    _, cube = model.generate_residuals(params + [1e3], return_datacube=True)
    spectrum = cube.sum(axis=(1, 2))
    w = model.imset.wavelength
    assert spectrum[np.argmin(np.abs(w - blue * (1 + ZS)))] > 0
    assert spectrum[np.argmin(np.abs(w - red * (1 + ZS)))] < 0
    # Mixed line types: white-light image is not flipped
    assert model.imset.whitelight_sign == 1.0


def test_unit_line_profile_ignores_slope_params(tmp_path):
    """init_spec_fit from adjust_slope=True carries trailing (m, b); they must not be read as ratios."""
    model, _ = _make_model(tmp_path, "absorption", restwave_peaks=TRIPLET, init_spec_fit=(ZS, 5.0, 1.0, 1.3, 1.5))
    z = np.full(4, ZS)
    sigma = np.full(4, 3.0)
    ref = model.unit_line_profile(z, sigma)

    model.imset.init_spec_fit = np.array([ZS, 5.0, 1.0, 1.3, 1.5, 0.1, 2.0])
    np.testing.assert_array_equal(model.unit_line_profile(z, sigma), ref)
    assert np.all(ref <= 0)


def test_vd_plots_label_absorbed_flux(tmp_path):
    import matplotlib.pyplot as plt

    model, params = _make_model(tmp_path, "absorption")
    plt.close("all")
    model.generate_residuals(params + [1e3], vd_plots=True)
    labels = [ax.get_ylabel() for ax in plt.gcf().axes]
    plt.close("all")
    assert "absorbed flux" in labels


def test_get_initial_spec_fit_absorption(trough_cube):
    """The 1D aperture fit recovers a positive depth for an absorption template."""
    cube, waves = trough_cube
    imset = ImageSet(cube, waves, ZS, 0.05, 1, 2)
    restwave_peaks = [1300.0, 1320.0]
    true_params = np.array([ZS + 0.001, 25.0, 80.0, 0.6])

    imset.restwave_peaks = restwave_peaks
    imset.line_types = "absorption"
    rng = np.random.default_rng(1)
    imset.aperature_spec = imset.gen_2d_spec(true_params) + rng.normal(0, 0.5, len(imset.wavelength))
    assert imset.aperature_spec.min() < -50

    imset.line_types = None  # set via get_initial_spec_fit below
    imset.get_initial_spec_fit(restwave_peaks, [ZS, 20.0, 60.0, 0.5], rewrite_zs=False, line_types="absorption")
    assert imset.whitelight_sign == -1.0
    np.testing.assert_allclose(imset.init_spec_fit, true_params, rtol=0.05)


def test_old_pickled_imageset_defaults_to_emission(trough_cube):
    """ImageSets pickled before line_types existed load and behave as emission-only."""
    import pickle

    cube, waves = trough_cube
    imset = ImageSet(cube, waves, ZS, 0.05, 1, 2)
    del imset.line_types
    imset = pickle.loads(pickle.dumps(imset))
    imset.restwave_peaks = [1300.0, 1320.0]

    assert np.array_equal(imset.line_signs, [1.0, 1.0])
    assert imset.whitelight_sign == 1.0
    assert imset.gen_2d_spec([ZS, 20.0, 10.0, 1.0]).max() > 0
