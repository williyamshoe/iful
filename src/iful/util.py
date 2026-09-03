"""
Utility functions for numerical analysis, geometry, kinematic profile evaluation, image masking, and MCMC post-processing.
"""

import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from PIL import Image
from lenstronomy.Util import param_util
import matplotlib.pyplot as plt
import math, os
from collections import Counter


def check_list(variable):
    """
    Check if a variable is a list or NumPy ndarray.

    Parameters
    ----------
    variable : object
        Variable to inspect.

    Returns
    -------
    bool
        True if variable is a list or numpy.ndarray, False otherwise.
    """
    if isinstance(variable, list):
        return True
    elif isinstance(variable, np.ndarray):
        return True
    else:
        return False


def check_bounds_proximity(params, lower_bounds, upper_bounds, atol=1e-3, rtol=1e-5):
    """
    Check whether parameter values are close to or outside lower/upper parameter bounds.

    Parameters
    ----------
    params : array-like
        Current parameter values.
    lower_bounds : array-like
        Lower bound constraints.
    upper_bounds : array-like
        Upper bound constraints.
    atol : float, default=1e-3
        Absolute tolerance margin.
    rtol : float, default=1e-5
        Relative tolerance margin.

    Returns
    -------
    dict
        Dictionary containing boolean arrays for violations and `is_safe` flag.
    """
    params = np.asarray(params)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    lower_margin = lower_bounds + atol + rtol * np.abs(lower_bounds)
    upper_margin = upper_bounds - (atol + rtol * np.abs(upper_bounds))
    at_or_past_lower = params <= lower_margin
    at_or_past_upper = params >= upper_margin
    any_violation = at_or_past_lower | at_or_past_upper

    return {
        "lower_violation": at_or_past_lower,
        "upper_violation": at_or_past_upper,
        "any_violation": any_violation,
        "is_safe": not any_violation.any() 
    }


def arctan_1d(A, B, r):
    """
    Evaluate 1D arctan velocity profile: A * arctan(B * r).

    Parameters
    ----------
    A : float
        Asymptotic velocity scale.
    B : float
        Turnover scale parameter.
    r : float or numpy.ndarray
        Distance from center along major axis.

    Returns
    -------
    float or numpy.ndarray
        Modeled velocity value(s).
    """
    return A * np.arctan(B * r)


def norm_dist(amp, mu, std, x):
    """
    Evaluate 1D unnormalized Gaussian distribution: amp * exp(-0.5 * ((x - mu)/std)^2).

    Parameters
    ----------
    amp : float
        Peak amplitude.
    mu : float
        Mean / center position.
    std : float
        Standard deviation (width).
    x : float or numpy.ndarray
        Evaluation coordinate(s).

    Returns
    -------
    float or numpy.ndarray
        Evaluated Gaussian profile value(s).
    """
    return amp * np.exp(-0.5 * (x - mu) ** 2 / std**2)


def distance_to_line(point, line_point, angle_degrees):
    """
    Calculate perpendicular distance from a point to a line specified by a point and position angle.

    Parameters
    ----------
    point : tuple of float
        Target coordinate (x1, y1).
    line_point : tuple of float
        Reference point on line (x0, y0).
    angle_degrees : float
        Position angle of the line in degrees.

    Returns
    -------
    float
        Perpendicular distance to the line.
    """
    x1, y1 = point
    x0, y0 = line_point
    theta = math.radians(angle_degrees)
    distance = (x1 - x0) * math.sin(theta) - (y1 - y0) * math.cos(theta)
    return distance


def arctan_2d(PA, A, B, C, c0, c1, r0, r1):
    """
    Evaluate 2D arctan velocity field given major axis position angle and center.

    Parameters
    ----------
    PA : float
        Kinematic position angle in degrees.
    A : float
        Asymptotic velocity amplitude.
    B : float
        Spatial scaling parameter.
    C : float
        Systemic velocity offset.
    c0, c1 : float
        Kinematic center coordinates (x0, y0).
    r0, r1 : float or numpy.ndarray
        Evaluation grid coordinates (x, y).

    Returns
    -------
    float or numpy.ndarray
        Evaluated 2D arctan velocity model values.
    """
    r = distance_to_line((r0, r1), (c0, c1), PA)
    return arctan_1d(A, B, r) + C


def tanh_1d(A, B, r):
    """
    Evaluate 1D tanh velocity profile: A * tanh(r / B).

    Parameters
    ----------
    A : float
        Asymptotic velocity scale.
    B : float
        Scale radius.
    r : float or numpy.ndarray
        Distance along major axis.

    Returns
    -------
    float or numpy.ndarray
        Modeled velocity profile.
    """
    B_safe = 1e-10 if B == 0 else B
    return A * np.tanh(r / B_safe)


def tanh_2d(PA, A, B, C, c0, c1, r0, r1):
    """
    Evaluate 2D tanh velocity field.

    Parameters
    ----------
    PA : float
        Kinematic position angle in degrees.
    A : float
        Asymptotic velocity amplitude.
    B : float
        Scale radius parameter.
    C : float
        Systemic velocity offset.
    c0, c1 : float
        Kinematic center (x0, y0).
    r0, r1 : float or numpy.ndarray
        Evaluation position(s).

    Returns
    -------
    float or numpy.ndarray
        Evaluated 2D tanh velocity field values.
    """
    r = distance_to_line((r0, r1), (c0, c1), PA)
    return tanh_1d(A, B, r) + C


def multiparam_1d(Vt, Rt, beta, xi, r):
    """
    Evaluate 1D multi-parameter rotation curve (Courteau / Catinella profile).

    Parameters
    ----------
    Vt : float
        Turnover velocity amplitude.
    Rt : float
        Turnover radius.
    beta : float
        Outer slope parameter.
    xi : float
        Inner sharpness parameter.
    r : float or numpy.ndarray
        Distance along major axis.

    Returns
    -------
    float or numpy.ndarray
        Evaluated velocity.
    """
    R = np.abs(r)
    R_safe = np.where(R == 0, 1e-10, R)
    xi_safe = 1e-10 if xi == 0 else xi
    val = Vt * ((1.0 + Rt / R_safe) ** beta) / ((1.0 + (Rt / R_safe) ** xi_safe) ** (1.0 / xi_safe))
    val = np.where(R == 0, 0.0, val)
    return np.sign(r) * val


def multiparam_2d(PA, Vt, Rt, beta, xi, C, c0, c1, r0, r1):
    """
    Evaluate 2D multi-parameter velocity field.

    Parameters
    ----------
    PA : float
        Kinematic position angle in degrees.
    Vt : float
        Turnover velocity.
    Rt : float
        Turnover radius.
    beta : float
        Outer slope index.
    xi : float
        Inner transition sharpness.
    C : float
        Systemic velocity offset.
    c0, c1 : float
        Center coordinates (x0, y0).
    r0, r1 : float or numpy.ndarray
        Evaluation grid coordinates.

    Returns
    -------
    float or numpy.ndarray
        Evaluated velocity field.
    """
    r = distance_to_line((r0, r1), (c0, c1), PA)
    return multiparam_1d(Vt, Rt, beta, xi, r) + C


def create_gif(image_paths, output_gif_path, duration=300, loop=0):
    """
    Assemble a sequence of saved PNG images into an animated GIF file.

    Parameters
    ----------
    image_paths : list of str
        Ordered file paths to input frame images.
    output_gif_path : str
        Output file path for generated GIF.
    duration : int, default=300
        Display duration per frame in milliseconds.
    loop : int, default=0
        Number of loops (0 means loop infinitely).
    """
    images = [Image.open(image_path) for image_path in image_paths]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
    )


def gen_gif(data_datacube, model_datacube, var_datacube, mask_3d, waves, name, overwrite=True, pixscale=None):
    """
    Generate an animated GIF showing data, model, and residual maps slice-by-slice along wavelength.

    Parameters
    ----------
    data_datacube : numpy.ndarray
        Observed 3D datacube (N_wave, N_x, N_y).
    model_datacube : numpy.ndarray
        Model 3D datacube matching data shape.
    var_datacube : numpy.ndarray
        Variance 3D datacube.
    mask_3d : numpy.ndarray
        3D binary mask cube.
    waves : numpy.ndarray
        1D wavelength array.
    name : str
        Output GIF filename.
    overwrite : bool, default=True
        Whether to overwrite existing file.
    pixscale : float, optional
        Pixel scale for spatial extent labeling (arcsec or degrees).
    """
    if os.path.isfile(name) and overwrite==False:
        return
    os.system("mkdir temp")
    
    if pixscale is not None:
        if pixscale < 1e-3:
            pixscale_arcsec = pixscale * 3600.0
        else:
            pixscale_arcsec = pixscale
        N_x, N_y = data_datacube.shape[1], data_datacube.shape[2]
        half_x = (N_x / 2.0) * pixscale_arcsec
        half_y = (N_y / 2.0) * pixscale_arcsec
        extent = [half_x, -half_x, -half_y, half_y]
        xlabel = "RA offset (arcsec)"
        ylabel = "DEC offset (arcsec)"
    else:
        extent = None
        xlabel = "Relative RA (pixels)"
        ylabel = "Relative DEC (pixels)"
        
    imfiles = []
    for i, _ in enumerate(waves):
        plt.figure(figsize=(13.5, 4))

        plt.subplot(1, 3, 1)
        plt.imshow((data_datacube * mask_3d)[i, :, :], vmin=0, vmax=30, extent=extent)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().invert_yaxis()
        plt.title(f"data")

        plt.subplot(1, 3, 2)
        plt.imshow((model_datacube * mask_3d)[i, :, :], vmin=0, vmax=30, extent=extent)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().invert_yaxis()
        plt.title(f"{waves[i]:.2f} Å\nmodel")

        plt.subplot(1, 3, 3)
        col = plt.imshow(
            ((data_datacube - model_datacube) * mask_3d / var_datacube**0.5)[i, :, :],
            vmin=-6,
            vmax=6,
            cmap="bwr",
            extent=extent,
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().invert_yaxis()
        plt.title(r"(data-model)/$\sigma$")
        plt.colorbar(col)

        plt.tight_layout()

        imfile = f"temp/{i}.png"
        plt.savefig(imfile, bbox_inches="tight")
        plt.clf()
        plt.close()
        imfiles += [imfile]
    create_gif(imfiles, name)
    os.system("rm -rf temp")


def mask_circle(x, y, rad, shape):
    """
    Create a 2D mask array with zeros inside a circle of given radius and center.

    Parameters
    ----------
    x, y : float
        Center pixel coordinates of circular region.
    rad : float
        Radius of circular region (in pixels).
    shape : tuple of int
        Output 2D array shape (N_row, N_col).

    Returns
    -------
    numpy.ndarray
        2D mask array with 0.0 inside circle and 1.0 outside.
    """
    mask = np.ones(shape, dtype=float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if ((i - x) ** 2 + (y - j) ** 2) ** 0.5 <= rad:
                mask[i][j] = 0.0
    return mask


def get_outlier_mask_iqr(data, scale_l=5, scale_u=30):
    """
    Compute a 2D boolean mask filtering out Interquartile Range (IQR) outliers.

    Parameters
    ----------
    data : numpy.ndarray
        2D input array.
    scale_l : float, default=5
        Lower IQR multiplier.
    scale_u : float, default=30
        Upper IQR multiplier.

    Returns
    -------
    numpy.ndarray
        2D boolean array (True for in-range pixels, False for outliers).
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    IQR = Q3 - Q1
    lower_bound = Q1 - scale_l * IQR
    upper_bound = Q3 + scale_u * IQR

    return (data > lower_bound) & (data < upper_bound)


def sum_of_gaussians(x, amplitudes, sigmas):
    """
    Sum multiple centered 1D Gaussians: sum(A_i * exp(-x^2 / (2 * sig_i^2))).

    Parameters
    ----------
    x : float or numpy.ndarray
        Evaluation coordinate.
    amplitudes : list of float
        Gaussian peak amplitudes.
    sigmas : list of float
        Gaussian widths.

    Returns
    -------
    float or numpy.ndarray
        Summed Gaussian profile value.
    """
    total_value = 0
    for A, sig in zip(amplitudes, sigmas):
        total_value += A * np.exp(-(x**2) / (2 * sig**2))
    return total_value


def calculate_fwhm(amplitudes, sigmas):
    """
    Calculate the Full Width at Half Maximum (FWHM) for a sum of centered Gaussians.

    Parameters
    ----------
    amplitudes : list of float
        Amplitudes of component Gaussians.
    sigmas : list of float
        Standard deviations of component Gaussians.

    Returns
    -------
    float
        Calculated FWHM value.
    """
    max_amplitude = sum_of_gaussians(0, amplitudes, sigmas)

    half_max_amplitude = max_amplitude / 2.0

    def find_half_max_x(x):
        return sum_of_gaussians(x, amplitudes, sigmas) - half_max_amplitude

    initial_guess = max(sigmas)

    hwhm = fsolve(find_half_max_x, initial_guess)[0]
    fwhm = 2 * hwhm
    return fwhm

def rename_repeats(strings):
    """
    Append numerical suffixes to repeated strings in a list (e.g. ['a', 'a'] -> ['a0', 'a1']).

    Parameters
    ----------
    strings : list of str
        Input list of string identifiers.

    Returns
    -------
    list of str
        List with repeated items indexed by numerical suffix.
    """
    counts = Counter(strings)
    
    seen_counts = {}
    result = []
    
    for s in strings:
        if counts[s] > 1:
            current_count = seen_counts.get(s, 0)
            result.append(f"{s}{current_count}")
            seen_counts[s] = current_count + 1
        else:
            result.append(s)
            
    return result

def homography_loss(points_src, points_dst):
    """
    Compute minimal squared distance loss between four source and target 2D points.

    Parameters
    ----------
    points_src : list or array-like
        4 source 2D coordinates.
    points_dst : list or array-like
        4 destination 2D coordinates.

    Returns
    -------
    float
        Total minimum squared assignment loss.
    """
    p_src = np.array(points_src)
    p_dst = np.array(points_dst)

    if p_src.shape != (4, 2) or p_dst.shape != (4, 2):
        raise ValueError("Both input lists must contain 5 2D points.")
    cost_matrix = np.sum(
        (p_src[:, np.newaxis, :] - p_dst[np.newaxis, :, :]) ** 2, axis=2
    )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    min_loss = cost_matrix[row_ind, col_ind].sum()

    return min_loss


def avg_dist(beta_ra, beta_dec, c_loc=None):
    """
    Compute mean Euclidean distance from points to mean position or reference location.

    Parameters
    ----------
    beta_ra : array-like
        RA coordinates.
    beta_dec : array-like
        Dec coordinates.
    c_loc : tuple of float, optional
        Reference position (ra_0, dec_0). Defaults to mean of inputs.

    Returns
    -------
    float
        Average distance.
    """
    if c_loc is None:
        mean_bra, mean_bdec = np.mean(beta_ra), np.mean(beta_dec)
    else:
        mean_bra, mean_bdec = c_loc[0], c_loc[1]
    dists = []
    for i in range(len(beta_ra)):
        dists += [((beta_ra - mean_bra) ** 2 + (beta_dec - mean_bdec) ** 2) ** 0.5]
    return np.mean(dists)


def min_total_squared_distance(observed_locations, predicted_locations):
    """
    Calculate minimal total squared assignment distance between observed and predicted image locations.

    Parameters
    ----------
    observed_locations : array-like
        Observed image coordinates.
    predicted_locations : array-like
        Model-predicted image coordinates.

    Returns
    -------
    float
        Minimum total squared distance.
    """
    A = np.array(observed_locations)
    B = np.array(predicted_locations)

    if len(A) == 0:
        return 0.0
    if len(B) == 0:
        return 1e7

    if len(B) < len(A):
        mean_point = np.mean(B, axis=0)
        points_needed = len(A) - len(B)
        padding = np.tile(mean_point, (points_needed, 1))
        B = np.vstack([B, padding])

    cost_matrix = cdist(A, B, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_min_sq_distance = cost_matrix[row_ind, col_ind].sum()

    return total_min_sq_distance


def distance_2d(x0, y0, x1, y1):
    """
    Compute 2D Euclidean distance between (x0, y0) and (x1, y1).

    Parameters
    ----------
    x0, y0 : float
        First point coordinates.
    x1, y1 : float
        Second point coordinates.

    Returns
    -------
    float
        Euclidean distance.
    """
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5


def least_squares_mean_loss(points_src, return_loc=False):
    """
    Compute sum of distances from 2D points to their mean location.

    Parameters
    ----------
    points_src : array-like
        Collection of 2D points.
    return_loc : bool, default=False
        If True, return mean location vector instead of loss value.

    Returns
    -------
    float or numpy.ndarray
        Distance loss value or mean location array.
    """
    p_src = np.array(points_src)
    mean_loc = np.mean(p_src, axis=0)
    if return_loc:
        return mean_loc
    loss = 0
    for p_ in p_src:
        loss += ((mean_loc[0] - p_[0]) ** 2 + (mean_loc[1] - p_[1]) ** 2) ** 0.5
    return loss


def check_within_bounds(inits, lowers, uppers):
    """
    Print warnings for initial parameter values outside specified lower and upper bounds.

    Parameters
    ----------
    inits : dict
        Initial parameter dictionary.
    lowers : dict
        Lower bounds dictionary.
    uppers : dict
        Upper bounds dictionary.
    """
    for k in inits:
        if inits[k] >= uppers[k] or inits[k] <= lowers[k]:
            print(f"{k} out of bounds")


def check_near_bounds(results, lowers, uppers, ind):
    """
    Print warnings if parameter fit results lie near boundary limits (within 1e-3).

    Parameters
    ----------
    results : dict
        Fit result parameters.
    lowers : dict
        Lower bounds dictionary.
    uppers : dict
        Upper bounds dictionary.
    ind : int or str
        Profile or component index identifier.
    """
    for k in results:
        if k in uppers and np.abs(results[k] - uppers[k]) <= 1e-3:
            print(
                f"profile {ind} parameter {k} near upper bound (value {results[k]}, bound {uppers[k]})"
            )
        elif k in lowers and np.abs(results[k] - lowers[k]) <= 1e-3:
            print(
                f"profile {ind} parameter {k} near lower bound (value {results[k]}, bound {lowers[k]})"
            )


def get_reduced_chi_sq(fitting_seq, kwargs_model, mask):
    """
    Calculate reduced chi-squared statistic using Lenstronomy ModelPlot.

    Parameters
    ----------
    fitting_seq : Lenstronomy FittingSequence
        Fitted sequence containing band list and best-fit parameters.
    kwargs_model : dict
        Model configuration dictionary.
    mask : numpy.ndarray
        2D likelihood mask array.

    Returns
    -------
    float
        Reduced chi-squared statistic.
    """
    lensPlot = ModelPlot(
        fitting_seq.multi_band_list,
        kwargs_model,
        fitting_seq.best_fit(),
        arrow_size=0.02,
        cmap_string="gist_heat",
        linear_solver=kwargs_constraints.get("linear_solver", True),
        image_likelihood_mask_list=np.array([mask]),
    )
    return lensPlot._band_plot_list[0].reduced_x2


def get_p_value(fitting_seq, kwargs_model, mask):
    """
    Calculate p-value for the reduced chi-squared fit statistic.

    Parameters
    ----------
    fitting_seq : Lenstronomy FittingSequence
        Fitting sequence.
    kwargs_model : dict
        Model dictionary.
    mask : numpy.ndarray
        2D binary mask.

    Returns
    -------
    float
        Survival function p-value.
    """
    chi_sq = get_reduced_chi_sq(fitting_seq, kwargs_model, mask)
    dof = int(np.sum(mask))
    return chi2.sf(chi_sq * dof, dof)

def prune_mcmc_chains(traces, deviation_threshold=3.5, stagnancy_threshold=0.01, split=False):
    """
    Identify and prune stagnant or outlier MCMC walker chains based on variance and median absolute deviation.

    Parameters
    ----------
    traces : numpy.ndarray
        MCMC walker chain array of shape (nwalkers, nsteps, nparams).
    deviation_threshold : float, default=3.5
        Modified Z-score threshold for identifying outlier walker means.
    stagnancy_threshold : float, default=0.01
        Relative variance threshold below which a walker is classified as stagnant.
    split : bool, default=False
        If True, return two flattened chain subsets (first third and last third of steps).

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Pruned walker chain array (or pair of split chain arrays if split=True).
    """
    traces = np.array(traces)
    C, L, P = traces.shape
    print(f"Processing MCMC output: {C} nwalkers, {L} steps, {P} parameters.")

    chain_variances = np.var(traces, axis=1)
    median_variances = np.median(chain_variances, axis=0)
    median_variances = np.where(median_variances == 0, 1e-12, median_variances)
    relative_variances = chain_variances / median_variances

    mean_relative_variance = np.mean(relative_variances, axis=1)
    stagnant_mask = mean_relative_variance < stagnancy_threshold
    chain_means = np.mean(traces, axis=1)

    grand_median = np.median(chain_means, axis=0)
    diff = np.abs(chain_means - grand_median)
    mad = np.median(diff, axis=0)

    mad = np.where(mad == 0, 1e-9, mad)
    modified_z_scores = 0.6745 * diff / mad
    max_z_per_chain = np.max(modified_z_scores, axis=1)
    outlier_mask = max_z_per_chain > deviation_threshold

    bad_chains_mask = stagnant_mask | outlier_mask
    kept_indices = np.where(~bad_chains_mask)[0]
    
    removed_count = C - len(kept_indices)
    if removed_count > 0:
        print(f"-> Pruned {removed_count} chains.")
        
    valid_traces = traces[kept_indices, :, :]

    if split:
        valid_traces_first = valid_traces[:, :L//3, :]
        valid_traces_second = valid_traces[:, -L//3:, :]
        return valid_traces_first.reshape(-1, P), valid_traces_second.reshape(-1, P)
        
    return valid_traces

def is_border_all_nan(arr_2d):
    """
    Check whether all perimeter border elements of a 2D array are NaN.

    Parameters
    ----------
    arr_2d : numpy.ndarray
        2D array to inspect.

    Returns
    -------
    bool
        True if all border elements are NaN, False otherwise.
    """
    if arr_2d.ndim != 2 or min(arr_2d.shape) < 1:
        return False
    nan_mask = np.isnan(arr_2d)

    top_border = nan_mask[0, :]
    bottom_border = nan_mask[-1, :]
    left_border = nan_mask[1:-1, 0]
    right_border = nan_mask[1:-1, -1]

    border_elements = np.concatenate(
        [top_border, bottom_border, left_border, right_border]
    )
    return np.all(border_elements)


def rotate_points(points, e1, e2):
    """
    Rotate 2D coordinates according to ellipticity parameters (e1, e2).

    Parameters
    ----------
    points : array-like
        Array of (x, y) coordinates.
    e1, e2 : float
        Ellipticity components.

    Returns
    -------
    numpy.ndarray
        Rotated (x, y) coordinates array.
    """
    angle, q = param_util.ellipticity2phi_q(e1, e2)
    angle = angle * -1
    rotated_points = []
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    for x, y in points:
        x_new = x * cos_angle - y * sin_angle
        y_new = x * sin_angle + y * cos_angle
        rotated_points.append((x_new, y_new))

    return np.array(rotated_points)


def find_closest_point_indices(points, target_points, threshold=2):
    """
    Find indices of nearest source grid points for each target coordinate within a radius threshold.

    Parameters
    ----------
    points : array-like
        Available grid point coordinates.
    target_points : array-like
        Target point coordinates to match.
    threshold : float, default=2
        Maximum allowed radial distance threshold. Target points beyond this return NaN.

    Returns
    -------
    list
        List of matching point indices (or NaN if outside threshold).
    """
    closest_indices = []
    for target_point in target_points:
        tx, ty = target_point

        if (tx**2 + ty**2) ** 0.5 > threshold:
            closest_indices.append(np.nan)
            continue

        closest_index = -1
        min_distance_sq = float("inf")
        for i, point in enumerate(points):
            px, py = point
            distance_sq = (px - tx) ** 2 + (py - ty) ** 2
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest_index = i
        closest_indices.append(int(closest_index))

    return closest_indices


def gen_obs_gif(data_datacube, mask_3d, waves, name, overwrite=True, spectrum=None, vmin=None, vmax=None, pixscale=None):
    """
    Generate an animated GIF showing datacube channel slices alongside the integrated spectrum.

    Parameters
    ----------
    data_datacube : numpy.ndarray
        3D observed datacube.
    mask_3d : numpy.ndarray
        3D binary mask.
    waves : numpy.ndarray
        1D wavelength array.
    name : str
        Output GIF filename.
    overwrite : bool, default=True
        Whether to overwrite existing GIF file.
    spectrum : numpy.ndarray, optional
        Custom 1D spectrum to display in side panel. Defaults to integrated flux spectrum.
    vmin, vmax : float, optional
        Display intensity range bounds.
    pixscale : float, optional
        Pixel scale for spatial extent labeling.
    """
    if os.path.isfile(name) and overwrite == False:
        return
    import shutil
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp", exist_ok=True)
    
    if pixscale is not None:
        if pixscale < 1e-3:
            pixscale_arcsec = pixscale * 3600.0
        else:
            pixscale_arcsec = pixscale
        N_x, N_y = data_datacube.shape[1], data_datacube.shape[2]
        half_x = (N_x / 2.0) * pixscale_arcsec
        half_y = (N_y / 2.0) * pixscale_arcsec
        extent = [half_x, -half_x, -half_y, half_y]
        xlabel = "RA offset (arcsec)"
        ylabel = "DEC offset (arcsec)"
    else:
        extent = None
        xlabel = "Relative RA (pixels)"
        ylabel = "Relative DEC (pixels)"
        
    if spectrum is None:
        spectrum = np.nansum(data_datacube * mask_3d, axis=(1, 2))
    if vmin is None:
        vmin = np.nanmin(data_datacube * mask_3d)
    if vmax is None:
        vmax = np.nanmax(data_datacube * mask_3d)
    imfiles = []
    for i, w in enumerate(waves):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow((data_datacube * mask_3d)[i, :, :], vmin=vmin, vmax=vmax, extent=extent)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().invert_yaxis()
        plt.title(f"data\n{w:.2f} Å")
        plt.subplot(1, 2, 2)
        plt.plot(waves, spectrum, color='blue', lw=2)
        plt.axvline(x=w, color='red', linestyle='--', lw=1.5)
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Relative Amplitude")
        plt.title("Spectrum")
        plt.tight_layout()
        imfile = f"temp/{i}.png"
        plt.savefig(imfile, bbox_inches="tight")
        plt.clf()
        plt.close()
        imfiles.append(imfile)
    create_gif(imfiles, name)
    shutil.rmtree("temp")

