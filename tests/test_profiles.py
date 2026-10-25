import pytest
import numpy as np

from iful.iful_modeling import IFULModel

G = 4.30241e-6  # (km/s)^2 kpc/M_sol, matches iful_modeling

@pytest.fixture
def aux_params():
    # [kwargs_source, sm, constant_val, d_s]
    kwargs_source = [{"center_x": 0.0, "center_y": 0.0, "e1": 0.0, "e2": 0.0}]
    return [kwargs_source, None, 50.0, 1000.0]

@pytest.fixture
def bare_model():
    # decide_profiles_fnc only needs the static methods, so skip __init__
    return object.__new__(IFULModel)

@pytest.mark.parametrize("name, expected_fnc, expected_n", [
    ("ARCTAN", "get_arctan_v_given_xy_bin", 4),
    ("SERSIC", "get_sersic_v_given_xy_bin", 1),
    ("GAUSSIAN", "get_gaussian_v_given_xy_bin", 2),
    ("POWER_LAW", "get_power_law_v_given_xy_bin", 2),
    ("EXPONENTIAL", "get_exponential_v_given_xy_bin", 3),
    ("POWER_LAW_BH", "get_power_law_bh_v_given_xy_bin", 3),
    ("CONSTANT_FIXED", "get_constant_v_given_xy_bin", 0),
    ("CONSTANT_FITTED", "get_constant_v_given_xy_bin", 1),
    ("CONSTANT_FITTED_BH", "get_constant_bh_v_given_xy_bin", 2),
    ("TANH", "get_tanh_v_given_xy_bin", 4),
    ("MULTIPARAM", "get_multiparam_v_given_xy_bin", 6),
    ("SHAPELETS_2", "get_shapelets_v_given_xy_bin", 6 + 1),
    ("SHAPELETS", "get_shapelets_v_given_xy_bin", 15 + 1),  # defaults to n_max=4
])
def test_decide_profiles_fnc(bare_model, name, expected_fnc, expected_n):
    fnc, n = bare_model.decide_profiles_fnc(name, num_bins=7)
    assert fnc is getattr(IFULModel, expected_fnc)
    assert n == expected_n

def test_decide_profiles_fnc_voronoi_and_unknown(bare_model):
    fnc, n = bare_model.decide_profiles_fnc("VORONOI", num_bins=7)
    assert fnc is IFULModel.get_voronoi_v_given_xy_bin
    assert n == 7

    with pytest.raises(Exception, match="Profile not implemented"):
        bare_model.decide_profiles_fnc("NOT_A_PROFILE", num_bins=7)

def test_exponential_profile(aux_params):
    central_vd, scale_rad, flat_vd = 100.0, 0.5, 30.0
    r = np.array([0.0, 0.5, 1.0, 50.0])
    vd = IFULModel.get_exponential_v_given_xy_bin(
        r, np.zeros_like(r), None, aux_params, [central_vd, scale_rad, flat_vd]
    )
    expected = np.sqrt(flat_vd**2 + central_vd**2 * np.exp(-0.5 * r**2 / scale_rad**2))
    assert np.allclose(vd, expected)
    # center adds in quadrature, far field falls to the flat floor
    assert np.isclose(vd[0], np.hypot(central_vd, flat_vd))
    assert np.isclose(vd[-1], flat_vd)
    assert np.all(np.diff(vd) <= 0)

def test_exponential_profile_requires_three_params(aux_params):
    with pytest.raises(ValueError):
        IFULModel.get_exponential_v_given_xy_bin(0.1, 0.1, None, aux_params, [100.0, 0.5])

def test_exponential_profile_circular_symmetry(aux_params):
    params = [100.0, 0.5, 30.0]
    a = IFULModel.get_exponential_v_given_xy_bin(0.3, 0.0, None, aux_params, params)
    b = IFULModel.get_exponential_v_given_xy_bin(0.0, -0.3, None, aux_params, params)
    assert np.allclose(a, b)

def test_power_law_profile(aux_params):
    r = np.array([0.5, 1.0, 2.0])
    vd = IFULModel.get_power_law_v_given_xy_bin(r, np.zeros_like(r), None, aux_params, [10.0, 1.0])
    assert np.allclose(vd, 10.0 * r**0.5)

def test_gaussian_profile(aux_params):
    r = np.array([0.0, 1.0])
    vd = IFULModel.get_gaussian_v_given_xy_bin(r, np.zeros_like(r), None, aux_params, [5.0, 1.0])
    assert np.allclose(vd, 5.0 * np.exp(-0.5 * r**2))

def test_constant_profiles(aux_params):
    x = np.array([0.1, 0.2, 0.3])
    fixed = IFULModel.get_constant_v_given_xy_bin(x, x, None, aux_params, [])
    assert np.allclose(fixed, aux_params[2])
    fitted = IFULModel.get_constant_v_given_xy_bin(x, x, None, aux_params, [80.0])
    assert np.allclose(fitted, 80.0)

def test_constant_bh_profile(aux_params):
    x, y = np.array([0.1, 1.0]), np.zeros(2)
    lg_bh_mass = 8.0
    dist = np.sqrt(x**2 + 1e-10)
    vd_bh_sq = G * 10**lg_bh_mass / (dist / 206265 * aux_params[3])

    # one fitted param: baseline comes from aux_params
    vd = IFULModel.get_constant_bh_v_given_xy_bin(x, y, None, aux_params, [lg_bh_mass])
    assert np.allclose(vd, np.sqrt(aux_params[2]**2 + vd_bh_sq))

    # two fitted params: baseline is fitted
    vd = IFULModel.get_constant_bh_v_given_xy_bin(x, y, None, aux_params, [70.0, lg_bh_mass])
    assert np.allclose(vd, np.sqrt(70.0**2 + vd_bh_sq))
    assert vd[0] > vd[1]
