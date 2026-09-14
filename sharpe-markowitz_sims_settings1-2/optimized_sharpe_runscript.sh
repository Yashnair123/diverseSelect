#!/usr/bin/bash
#SBATCH --job-name=clusterSim
#SBATCH --array=0-14
#SBATCH --output=outputs/opt_sharpe_%A_%a.out
#SBATCH --error=errors/opt_sharpe_%A_%a.err
#SBATCH --time=0-06:00
#SBATCH -p candes,stat,normal,owners,hns
#SBATCH -c 6
#SBATCH --mem=10GB


# Keep bash startup clean; initialize conda explicitly
source /home/users/yashnair/miniconda3/etc/profile.d/conda.sh
conda activate yash310

# Load compiler so the compiled extension & C++ runtime are available
ml gcc/9

# MOSEK license
export MOSEKLM_LICENSE_FILE=$SCRATCH/mosek/mosek.lic

# Use your packed env Python
python optimized_sharpe_get_histogram.py ${SLURM_ARRAY_TASK_ID}