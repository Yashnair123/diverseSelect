#!/usr/bin/bash
#SBATCH --job-name=clusterSim
#SBATCH --array=0-239
#SBATCH --output=outputs/job_%A_%a.out
#SBATCH --error=errors/job_%A_%a.err
#SBATCH --time=0-02:00
#SBATCH -p candes,stat,normal
#SBATCH -c 1
#SBATCH --mem=10GB

conda run -n yash python3 sherlock_cluster_driver.py ${SLURM_ARRAY_TASK_ID}