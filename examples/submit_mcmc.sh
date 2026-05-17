#!/bin/bash
#SBATCH --job-name=iful_mcmc
#SBATCH --qos=preempt
#SBATCH --constraint=cpu
#SBATCH --nodes=2                
#SBATCH --ntasks=128             # 1 master + 127 walkers
#SBATCH --cpus-per-task=2        # Give each task 2 physical cores
#SBATCH --time=12:00:00           
#SBATCH --output=mcmc_run_%j.log
#SBATCH --error=mcmc_err_%j.log
#SBATCH --account=deepsrch 

module load python
conda activate iful

# Define the number of threads to match --cpus-per-task
export THREADS=2

# Multithreading for Numpy/Scipy
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

# Multithreading for Numba (picked up by your python script)
export NUMBA_NUM_THREADS=$THREADS

# Launch the hybrid job!
srun -n 128 --cpus-per-task=$THREADS python run_mcmc_mpi.py