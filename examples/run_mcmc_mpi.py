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

def log_prob(params, priors=[(6, 1.104, 0.025), (7, 1.588, 0.041)]):
    # We must access ifulmodel4 and bounds from the global scope defined by the master
    global ifulmodel4, iful_lowerbounds, iful_upperbounds
    
    if not (np.all(params >= iful_lowerbounds) and np.all(params <= iful_upperbounds)):
        return -np.inf
    
    chi2_data = ifulmodel4.generate_residuals(params, linear_solve=True)
    chi2_prior = 0.0
    if priors is not None:
        for idx, mu, sigma in priors:
            chi2_prior += ((params[idx] - mu) / sigma) ** 2
            
    return -0.5 * (chi2_data + chi2_prior)

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
    iful_profiles = ["ARCTAN", "POWER_LAW_BH", "VORONOI"]
    ifulmodel4 = IFULModel(
        imset4, flatmodel4, iful_profiles,
        sourceplane_size=100, num_bins=50, num_rsersics=3,
        spectral_res=3500, equal_weight_voronoi=False, d_s=d_s,
    )

    # Reconstruct bounds
    lensing_lower_bounds, lensing_upper_bounds = ifulmodel4.init_fitting_seq.likelihoodModule.param_limits
    base_lower = [0, 0, 0, 1.430 * c, 1, 1.0, 4]
    base_upper = [360, 1000, 10, 1.436 * c, 500, 3.0, 12]
    
    global iful_lowerbounds, iful_upperbounds
    iful_lowerbounds = np.array(list(lensing_lower_bounds) + base_lower)
    iful_upperbounds = np.array(list(lensing_upper_bounds) + base_upper)

    # Load the best PSO result to use as our MCMC initial position
    iful_pso_results_filename = "s4_models/iful_pso_results_params.pickle"
    with open(iful_pso_results_filename, "rb") as handle:
        previous_results = pickle.load(handle)
        
    res_key = "_".join(iful_profiles) + "_ifulall"
    init_params = previous_results[res_key]

# ==========================================================================
    # 2. HYPERPARAMETERS
    # ==========================================================================
    ndim = len(init_params)
    mcmc_nwalkers = 127
    param_names = [f"param_{i}" for i in range(ndim)]

    run_comp_len = 100
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
            init_pos_mcmc = np.array(init_params) + 1e-4 * np.random.randn(mcmc_nwalkers, ndim)
            pos = np.clip(init_pos_mcmc, iful_lowerbounds + 1e-8, iful_upperbounds - 1e-8)
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
            pool=pool, backend=backend, moves=moves
        )

        while True:
            if not converged_b4:
                bandend_i += 1
                print(f"\n--- Starting run chunk {bandend_i} ---")

            if converged_b4 or convergence:
                os.system(f"touch {model_dir}/CONVERGED.txt")
                print("Chains have converged! Exiting loop.")
                break

            sampler.run_mcmc(pos, run_comp_len, progress=True)
            pos = None 

            with open(f'{model_dir}/bandend_i.txt', "w") as f:
                f.write(f"{bandend_i}")

            total_steps = sampler.iteration
            print(f"Total accumulated iterations in backend: {total_steps}")

            tau = sampler.get_autocorr_time(tol=0)
            max_tau = np.max(tau)
            
            converged = np.all(tau * 20 < total_steps)
            if not np.isinf(old_tau).any():
                tau_diff = np.abs(old_tau - tau) / tau
                converged &= np.all(tau_diff < 0.05)
            else:
                converged = False
                
            old_tau = tau

            print(f"Max autocorrelation time (tau): {max_tau:.2f}")
            print(f"Current iterations: {total_steps} (Target for convergence: > {20 * max_tau:.2f})")
            print(f"CONVERGENCE : {converged}")
            
            burnin = int(2 * max_tau) if max_tau > 0 else 0
            if burnin >= total_steps:
                burnin = total_steps // 2

            print(f"Discarding {burnin} steps as burn-in for summary statistics...")
            
            try:
                flat_chain = sampler.get_chain(discard=burnin, flat=True)
                medians = np.median(flat_chain, axis=0)
                stds = np.std(flat_chain, axis=0)

                for i, (p, med, mstd) in enumerate(zip(param_names, medians, stds)):
                    conv_flag = (tau[i] * 20 < total_steps)
                    print(f"{p:<25}: {med:>15.5f} +- {mstd:>15.5f}    tau: {tau[i]:>6.1f}   conv: {conv_flag}")
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