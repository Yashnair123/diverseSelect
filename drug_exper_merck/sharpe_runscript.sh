#!/usr/bin/bash
#SBATCH --job-name=clusterSim
#SBATCH --array=0-249
#SBATCH --output=outputs/sharpe_%A_%a.out
#SBATCH --error=errors/sharpe_%A_%a.err
#SBATCH --time=0-20:00
#SBATCH -p candes,normal,owners
#SBATCH -c 6
#SBATCH -C CPU_MNF:INTEL
#SBATCH --mem=10GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yashnair@stanford.edu


# Keep bash startup clean; initialize conda explicitly
source /home/users/yashnair/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/conda-envs/yash310

# Load compiler so the compiled extension & C++ runtime are available
ml gcc/9

# MOSEK license
export MOSEKLM_LICENSE_FILE=$SCRATCH/mosek/mosek.lic

# Use your packed env Python
python sherlock_sharpe_driver.py ${SLURM_ARRAY_TASK_ID}