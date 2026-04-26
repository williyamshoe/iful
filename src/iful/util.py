import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from PIL import Image
from IPython.display import Image as imp
from lenstronomy.Util import param_util
import matplotlib.pyplot as plt
import math, os


def check_list(variable):
    if isinstance(variable, list):
        return True
    elif isinstance(variable, np.ndarray):
        return True
    else:
        return False


def check_bounds_proximity(params, lower_bounds, upper_bounds, atol=1e-3, rtol=1e-5):
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
    return A * np.arctan(B * r)


def norm_dist(amp, mu, std, x):
    return amp * np.exp(-0.5 * (x - mu) ** 2 / std**2)


def distance_to_line(point, line_point, angle_degrees):
    x1, y1 = point
    x0, y0 = line_point
    theta = math.radians(angle_degrees)
    distance = (x1 - x0) * math.sin(theta) - (y1 - y0) * math.cos(theta)
    return distance


def arctan_2d(PA, A, B, C, c0, c1, r0, r1):
    r = distance_to_line((r0, r1), (c0, c1), PA)
    return arctan_1d(A, B, r) + C


def create_gif(image_paths, output_gif_path, duration=300, loop=0):
    images = [Image.open(image_path) for image_path in image_paths]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
    )


def gen_gif(data_datacube, model_datacube, var_datacube, mask_3d, waves, name):
    os.system("mkdir temp")
    imfiles = []
    for i, _ in enumerate(waves):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow((data_datacube * mask_3d)[i, :, :], vmin=0, vmax=30)
        plt.gca().set_axis_off()
        plt.gca().invert_yaxis()
        plt.title(f"data")

        plt.subplot(1, 3, 2)
        plt.imshow((model_datacube * mask_3d)[i, :, :], vmin=0, vmax=30)
        plt.gca().set_axis_off()
        plt.gca().invert_yaxis()
        plt.title(f"{waves[i]:.2f} Å\nmodel")

        plt.subplot(1, 3, 3)
        plt.imshow(
            ((data_datacube - model_datacube) * mask_3d / var_datacube**0.5)[i, :, :],
            vmin=-6,
            vmax=6,
            cmap="bwr",
        )
        plt.gca().set_axis_off()
        plt.gca().invert_yaxis()
        plt.title(f"data-model")

        plt.tight_layout()

        imfile = f"temp/{i}.png"
        plt.savefig(imfile, bbox_inches="tight")
        plt.clf()
        plt.close()
        imfiles += [imfile]
    create_gif(imfiles, name)
    os.system("rm -rf temp")


def mask_circle(x, y, rad, shape):
    mask = np.ones(shape, dtype=float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if ((i - x) ** 2 + (y - j) ** 2) ** 0.5 <= rad:
                mask[i][j] = 0.0
    return mask


def get_outlier_mask_iqr(data, scale_l=5, scale_u=30):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    IQR = Q3 - Q1
    lower_bound = Q1 - scale_l * IQR
    upper_bound = Q3 + scale_u * IQR

    return (data > lower_bound) & (data < upper_bound)


def norm_dist(amp, mu, std, x):
    return amp * np.exp(-0.5 * (x - mu) ** 2 / std**2)


def sum_of_gaussians(x, amplitudes, sigmas):
    total_value = 0
    for A, sig in zip(amplitudes, sigmas):
        total_value += A * np.exp(-(x**2) / (2 * sig**2))
    return total_value


def calculate_fwhm(amplitudes, sigmas):
    max_amplitude = sum_of_gaussians(0, amplitudes, sigmas)

    half_max_amplitude = max_amplitude / 2.0

    def find_half_max_x(x):
        return sum_of_gaussians(x, amplitudes, sigmas) - half_max_amplitude

    initial_guess = max(sigmas)

    hwhm = fsolve(find_half_max_x, initial_guess)[0]
    fwhm = 2 * hwhm
    return fwhm


def homography_loss(points_src, points_dst):
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
    if c_loc is None:
        mean_bra, mean_bdec = np.mean(beta_ra), np.mean(beta_dec)
    else:
        mean_bra, mean_bdec = c_loc[0], c_loc[1]
    dists = []
    for i in range(len(beta_ra)):
        dists += [((beta_ra - mean_bra) ** 2 + (beta_dec - mean_bdec) ** 2) ** 0.5]
    return np.mean(dists)


def min_total_squared_distance(observed_locations, predicted_locations):
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
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5


def least_squares_mean_loss(points_src, return_loc=False):
    p_src = np.array(points_src)
    mean_loc = np.mean(p_src, axis=0)
    if return_loc:
        return mean_loc
    loss = 0
    for p_ in p_src:
        loss += ((mean_loc[0] - p_[0]) ** 2 + (mean_loc[1] - p_[1]) ** 2) ** 0.5
    return loss


def check_within_bounds(inits, lowers, uppers):
    for k in inits:
        if inits[k] >= uppers[k] or inits[k] <= lowers[k]:
            print(f"{k} out of bounds")


def check_near_bounds(results, lowers, uppers, ind):
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
    chi_sq = get_reduced_chi_sq(fitting_seq, kwargs_model, mask)
    dof = int(np.sum(mask))
    return chi2.sf(chi_sq * dof, dof)


def prune_mcmc_chains(
    traces, deviation_threshold=3.5, stagnancy_threshold=0.01, split=False
):
    traces = np.array(traces)

    C, L, P = traces.shape

    print(f"Processing MCMC output: {C} chains, {L} iterations, {P} parameters.")

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
        print(f"   - Stagnant (low relative var): {np.sum(stagnant_mask)}")
        print(f"   - Outliers (bad convergence): {np.sum(outlier_mask)}")

    valid_traces = traces[kept_indices, :, :]

    if split:
        valid_traces_first = valid_traces[:, : L // 2, :]
        valid_traces_second = valid_traces[:, L // 2 :, :]

        flattened_traces_first = valid_traces_first.reshape(-1, P)
        flattened_traces_second = valid_traces_second.reshape(-1, P)

        return flattened_traces_first, flattened_traces_second

    return valid_traces


def is_border_all_nan(arr_2d):
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
