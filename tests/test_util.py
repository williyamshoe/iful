import pytest
import numpy as np
from iful.util import (
    check_list,
    check_bounds_proximity,
    distance_to_line,
    arctan_1d,
    arctan_2d,
    tanh_1d,
    tanh_2d,
    multiparam_1d,
    multiparam_2d,
    mask_circle,
    get_outlier_mask_iqr,
    norm_dist,
    sum_of_gaussians,
    calculate_fwhm,
    rename_repeats,
    homography_loss,
    avg_dist,
    min_total_squared_distance,
    distance_2d,
    least_squares_mean_loss,
    prune_mcmc_chains,
    is_border_all_nan,
    rotate_points,
    find_closest_point_indices,
)

def test_check_list():
    assert check_list([1, 2, 3]) is True
    assert check_list(np.array([1, 2, 3])) is True
    assert check_list("not a list") is False
    assert check_list(42) is False

def test_check_bounds_proximity():
    params = [1.0, 5.0, 9.999]
    lowers = [0.0, 0.0, 0.0]
    uppers = [10.0, 10.0, 10.0]
    res = check_bounds_proximity(params, lowers, uppers, atol=1e-2)
    assert res["is_safe"] is False
    assert res["upper_violation"][2] == True
    assert res["lower_violation"].any() == False

def test_distance_to_line():
    # Distance from point (0, 1) to horizontal line passing through (0, 0) at angle 0
    # line angle 0 => theta=0 => sin(0)=0, cos(0)=1 => (x1-x0)*0 - (y1-y0)*1 = -1
    d = distance_to_line((0, 1), (0, 0), 0)
    assert np.isclose(d, -1.0)

def test_arctan_and_tanh_profiles():
    # 1D arctan
    val = arctan_1d(A=2.0, B=1.0, r=1.0)
    assert np.isclose(val, 2.0 * np.arctan(1.0))

    # 2D arctan
    val2d = arctan_2d(PA=0, A=2.0, B=1.0, C=0.5, c0=0, c1=0, r0=0, r1=1)
    assert np.isclose(val2d, 2.0 * np.arctan(-1.0) + 0.5)

    # 1D tanh
    val_tanh = tanh_1d(A=2.0, B=1.0, r=1.0)
    assert np.isclose(val_tanh, 2.0 * np.tanh(1.0))

    # 2D tanh
    val2d_tanh = tanh_2d(PA=0, A=2.0, B=1.0, C=0.0, c0=0, c1=0, r0=0, r1=1)
    assert np.isclose(val2d_tanh, 2.0 * np.tanh(-1.0))

def test_multiparam_1d_and_2d():
    val = multiparam_1d(Vt=100.0, Rt=1.0, beta=0.5, xi=1.0, r=2.0)
    assert val > 0
    val2d = multiparam_2d(PA=0, Vt=100.0, Rt=1.0, beta=0.5, xi=1.0, C=0.0, c0=0, c1=0, r0=2.0, r1=0.0)
    assert isinstance(val2d, float)

def test_mask_circle():
    mask = mask_circle(x=5, y=5, rad=2.0, shape=(10, 10))
    assert mask.shape == (10, 10)
    assert mask[5, 5] == 0.0
    assert mask[0, 0] == 1.0

def test_get_outlier_mask_iqr():
    data = np.array([1, 2, 2, 3, 2, 2, 3, 100])
    outliers_mask = get_outlier_mask_iqr(data, scale_l=1.5, scale_u=1.5)
    assert outliers_mask[-1] == False
    assert outliers_mask[0] == True

def test_norm_dist_and_gaussians():
    val = norm_dist(amp=1.0, mu=0.0, std=1.0, x=0.0)
    assert np.isclose(val, 1.0)

    fwhm = calculate_fwhm(amplitudes=[1.0], sigmas=[1.0])
    assert np.isclose(fwhm, 2.354820, atol=1e-3)

def test_rename_repeats():
    names = ["a", "b", "a", "c", "a"]
    renamed = rename_repeats(names)
    assert renamed == ["a0", "b", "a1", "c", "a2"]

def test_homography_loss():
    pts1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    pts2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    loss = homography_loss(pts1, pts2)
    assert np.isclose(loss, 0.0)

def test_distances_and_losses():
    assert np.isclose(distance_2d(0, 0, 3, 4), 5.0)

    points = [[0, 0], [2, 0]]
    loss = least_squares_mean_loss(points)
    assert np.isclose(loss, 2.0)
    loc = least_squares_mean_loss(points, return_loc=True)
    assert np.allclose(loc, [1.0, 0.0])

    a = [[0, 0], [1, 1]]
    b = [[0, 0], [1, 1]]
    min_dist = min_total_squared_distance(a, b)
    assert np.isclose(min_dist, 0.0)

def test_prune_mcmc_chains():
    # 5 walkers, 100 steps, 2 params
    np.random.seed(42)
    chains = np.random.normal(loc=0.0, scale=1.0, size=(5, 100, 2))
    # Add one extreme outlier chain
    chains[4] += 100.0

    valid = prune_mcmc_chains(chains, deviation_threshold=3.0)
    assert valid.shape[0] < 5
    assert valid.shape[1] == 100
    assert valid.shape[2] == 2

def test_is_border_all_nan():
    arr = np.full((5, 5), np.nan)
    arr[2, 2] = 1.0
    assert is_border_all_nan(arr) == True

    arr[0, 0] = 1.0
    assert is_border_all_nan(arr) == False

def test_rotate_points():
    pts = np.array([[1.0, 0.0]])
    # e1=0, e2=0 => angle 0
    rotated = rotate_points(pts, e1=0.0, e2=0.0)
    assert np.allclose(rotated, [[1.0, 0.0]])

def test_find_closest_point_indices():
    points = [[0, 0], [1, 1], [2, 2]]
    targets = [[0.1, 0.1], [5, 5]]
    indices = find_closest_point_indices(points, targets, threshold=3.0)
    assert indices[0] == 0
    assert np.isnan(indices[1])
