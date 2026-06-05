import sys
import os
import time
import pickle
import numpy as np
import emcee
from schwimmbad import MPIPool

numba_threads = os.environ.get('NUMBA_NUM_THREADS', '1')
os.environ['NUMBA_NUM_THREADS'] = numba_threads

# Now import numba and set its thread count
import numba
numba.set_num_threads(int(numba_threads))

sys.path.append("../src")

# IFUL and Lenstronomy imports
from iful.util import *
from iful.image_set import *
from iful.flat_modeling import *
from iful.iful_modeling import *
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

def log_prob(params, priors=[(7, 1.588, 0.041)], lambda_reg=1e4):
    params = list(params[:6]) + [0] + list(params[6:])
    
    if not (np.all(params >= iful_lowerbounds) and np.all(params <= iful_upperbounds)):
        return -np.inf

    params[-3] += mean_velocity
    
    # Set return_datacube=True to extract the linear inversion flux parameters
    output = ifulmodel4.generate_residuals(params, linear_solve=True, return_datacube=True)
    chi2_data = output[0]
    flx_params = output[2] 
    
    # ==========================================
    # THE ROUGHNESS PENALTY
    # ==========================================
    # Check if we are using Shapelets or spatial bins (Voronoi/Pixels)
    if ifulmodel4.iful_profiles[-1].startswith("SHAPELETS"):
        # For shapelets, flx_params[0] is beta. flx_params[1] is the base Gaussian.
        # We only want to penalize the high-frequency higher-order amplitudes.
        high_order_amps = flx_params[2:] 
        reg_penalty = lambda_reg * np.sum(high_order_amps**2)
    else:
        # For Voronoi or grid bins, penalize the variance (spikiness) of the fluxes.
        # A perfectly smooth, bright source has a variance of 0.
        reg_penalty = 0.0 #lambda_reg * np.var(flx_params)
        
    chi2_prior = 0.0
    if priors is not None:
        for idx, mu, sigma in priors:
            chi2_prior += ((params[idx] - mu) / sigma) ** 2
            
    # Add the regularization penalty
    return -0.5 * (chi2_data + reg_penalty + chi2_prior)

def main():
    # ==========================================================================
    # 1. LOAD PRE-COMPUTED DATA & SET UP MODEL
    # ==========================================================================
    # Wait for flat chains if necessary
    while not os.path.exists("s4_models/flat_chains.pickle"):
        time.sleep(60)

    with open("s4_models/imset4.pickle", "rb") as handle:
        imset4 = pickle.load(handle)

    with open("s4_models/flatmodel4.pickle", "rb") as handle:
        flatmodel4 = pickle.load(handle)

    c = 299792
    d_s = FlatLambdaCDM(H0=70, Om0=0.3).angular_diameter_distance(imset4.zs).to(u.kpc).value

    global ifulmodel4
    iful_profiles = ["ARCTAN", "CONSTANT_FITTED_BH", "VORONOI"]
    ifulmodel4 = IFULModel(
        imset4, flatmodel4, iful_profiles,
        sourceplane_size=100, num_bins=50, num_rsersics=3,
        spectral_res=3500, equal_weight_voronoi=False, d_s=d_s,
    )

    # Reconstruct bounds
    lensing_lower_bounds, lensing_upper_bounds = ifulmodel4.init_fitting_seq.likelihoodModule.param_limits
    base_lower = [0, 0, 0, 1.430 * c, 0., 5.0]
    base_upper = [360, 1000, 10, 1.436 * c, 300, 10.0]
    
    global iful_lowerbounds, iful_upperbounds
    iful_lowerbounds = np.array(list(lensing_lower_bounds) + base_lower)
    iful_upperbounds = np.array(list(lensing_upper_bounds) + base_upper)

    # Load the best PSO result to use as our MCMC initial position
    iful_pso_results_filename = "s4_models/iful_pso_results_params.pickle"
    with open(iful_pso_results_filename, "rb") as handle:
        previous_results = pickle.load(handle)
        
    res_key = "_".join(iful_profiles) + "_ifulall"
    init_params = previous_results[res_key]

    global mean_velocity
    mean_velocity = init_params[-3]

    init_params = np.array(list(init_params[:-3]) + [init_params[-3] - mean_velocity] + list(init_params[-2:]))
    iful_lowerbounds = np.array(list(iful_lowerbounds[:-3]) + [iful_lowerbounds[-3] - mean_velocity] + list(iful_lowerbounds[-2:]))
    iful_upperbounds = np.array(list(iful_upperbounds[:-3]) + [iful_upperbounds[-3] - mean_velocity] + list(iful_upperbounds[-2:]))

    # ==========================================================================
    # 2. HYPERPARAMETERS
    # ==========================================================================
    ndim = len(init_params) - 1
    mcmc_nwalkers = 127
    param_names = [f"param_{i}" for i in range(ndim)]

    run_comp_len = 300
    max_iterations = 100
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]

    # ==========================================================================
    # 3. MPI EXECUTION & MASTER-ONLY SETUP
    # ==========================================================================
    with MPIPool() as pool:
        # ----------------------------------------------------------------------
        # WORKER NODES: Stop here and wait for instructions to evaluate log_prob
        # ----------------------------------------------------------------------
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        # ----------------------------------------------------------------------
        # MASTER NODE: Handles all file I/O, backend setup, and sampler management
        # ----------------------------------------------------------------------
        model_dir = "s4_models"
        os.makedirs(model_dir, exist_ok=True)
        
        bf = f"{model_dir}/model_backup.hdf5"
        first = not os.path.isfile(bf)
        converged_b4 = os.path.isfile(f'{model_dir}/CONVERGED.txt')
        convergence = False

        if not os.path.isfile(f"{model_dir}/bandend_i.txt"):
            with open(f'{model_dir}/bandend_i.txt', "w") as f:
                f.write("0")
                
        with open(f'{model_dir}/bandend_i.txt') as f:
            bandend_i = int(f.readlines()[-1])

        backend = emcee.backends.HDFBackend(bf, name="custom_mcmc_emcee")

        if first or backend.iteration == 0: 
            print("\n=======================================================")
            print(" INITIALIZING NEW MCMC CHAIN FROM SCRATCH")
            print("=======================================================\n")
            backend.reset(mcmc_nwalkers, ndim)
        
            init_pos_mcmc = [init_params]
            while len(init_pos_mcmc) < mcmc_nwalkers:
                scale = (iful_upperbounds - iful_lowerbounds) * 0.01
                all_init_fits_t = np.array(init_params) + np.random.normal(0, scale)
                if np.all(all_init_fits_t >= iful_lowerbounds) and np.all(all_init_fits_t <= iful_upperbounds):
                    init_pos_mcmc += [all_init_fits_t]
            pos = np.array(init_pos_mcmc)
            pos = np.delete(pos, 6, axis=1)
            
        else:
            print("\n=======================================================")
            print(f" RESUMING INTERRUPTED RUN!")
            print(f" Found {backend.iteration} completed steps in {bf}.")
            print("=======================================================\n")
            pos = backend.get_last_sample()
            
        old_tau = np.inf

        # Initialize the native emcee sampler
        sampler = emcee.EnsembleSampler(
            mcmc_nwalkers, ndim, log_prob, 
            pool=pool, backend=backend, moves=moves,
        )

        while True:
            if not converged_b4:
                bandend_i += 1
                print(f"\n--- Starting run chunk {bandend_i} ---")

            if converged_b4 or convergence:
                os.system(f"touch {model_dir}/CONVERGED.txt")
                print("Chains have converged! Exiting loop.")
                break

            sampler.run_mcmc(pos, run_comp_len//3, progress=True)
            pos = None 

            with open(f'{model_dir}/bandend_i.txt', "w") as f:
                f.write(f"{bandend_i}")

            total_steps = sampler.iteration
            print(f"Total accumulated iterations in backend: {total_steps}")

            full_chain = sampler.get_chain(flat=False)

            burnin_cutoff = -1 * run_comp_len
            gl_stat = full_chain[burnin_cutoff:, :, :]

            if len(gl_stat) < run_comp_len:
                continue
                
            gl_stat = gl_stat.transpose(1, 0, 2)
            pruned_collapsed_chains_first, pruned_collapsed_chains_second = prune_mcmc_chains(gl_stat, split=True)
        
            first_16 = np.percentile(pruned_collapsed_chains_first, 16, axis=0)
            first_84 = np.percentile(pruned_collapsed_chains_first, 84, axis=0)
        
            second_16 = np.percentile(pruned_collapsed_chains_second, 16, axis=0)
            second_84 = np.percentile(pruned_collapsed_chains_second, 84, axis=0)
            allowed_var = (second_84 - second_16)*.05
        
            second_50 = np.percentile(pruned_collapsed_chains_second, 50, axis=0)
            second_std = np.std(pruned_collapsed_chains_second, axis=0)
        
            convergence_ind = [(np.abs(s16 - f16) <= av) and (np.abs(s84 - f84) <= av) for f16, f84, s16, s84, av in zip(first_16, first_84, second_16, second_84, allowed_var)]
            converged = np.sum(convergence_ind)/len(convergence_ind) >= 0.9

            # tau = sampler.get_autocorr_time(tol=0)
            # max_tau = np.max(tau)
            
            # converged = np.all(tau * 20 < total_steps)
            # if not np.isinf(old_tau).any():
            #     tau_diff = np.abs(old_tau - tau) / tau
            #     converged &= np.all(tau_diff < 0.05)
            # else:
            #     converged = False
                
            # old_tau = tau

            # print(f"Max autocorrelation time (tau): {max_tau:.2f}")
            # print(f"Current iterations: {total_steps} (Target for convergence: > {20 * max_tau:.2f})")
            # print(f"CONVERGENCE : {converged}")
            
            # burnin = int(2 * max_tau) if max_tau > 0 else 0
            # if burnin >= total_steps:
            #     burnin = total_steps // 2
            
            try:

                for i, (p, med, mstd, conv_flag) in enumerate(zip(param_names, second_50, second_std, convergence_ind)):
                    print(f"{p:<25}: {med:>15.5f} +- {mstd:>15.5f}   conv: {conv_flag}")
            except ValueError:
                print("Chain still too short to compute reliable summary statistics.")

            if converged:
                convergence = True
            elif bandend_i >= max_iterations:
                os.system(f"touch {model_dir}/fail_to_converge.txt")
                print("Reached maximum chunks without converging. Exiting loop.")
                break

if __name__ == "__main__":
    main()