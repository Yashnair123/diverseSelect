import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

given_setting = int(sys.argv[1])
variant_indices = []
variants = []
variant_counter = 0
for job in range(250):
  for setting_ind in range(2):
    for alpha_ind in range(3):
      variants.append((job, setting_ind, alpha_ind))
      if setting_ind == given_setting:
        variant_indices.append(variant_counter)
      variant_counter += 1
dne = 0
vanilla_dne = 0
for variant in tqdm(variant_indices):
    (job, setting, alpha_ind) = variants[variant]
    
    try:
      true_true_diversity_results \
        = pd.read_csv(f'sharpe_results/metrics_v{variant}_w{True}_c{True}.csv', header=None).to_numpy().astype(float)[0]
      true_true_diversity_time_results \
        = pd.read_csv(f'sharpe_results/solver_times_v{variant}_w{True}_c{True}.csv', header=None).to_numpy().astype(float)[0]
      

      with open(f"collated_sharpe_results/metrics_s{setting}_a{alpha_ind}_w{True}_c{True}.csv", "at") as file:
        file.write(",".join(map(str, true_true_diversity_results)) + "\n")

      with open(f"collated_sharpe_results/solver_times_s{setting}_a{alpha_ind}_w{True}_c{True}.csv", "at") as file:
        file.write(",".join(map(str, true_true_diversity_time_results)) + "\n")
      
      false_true_diversity_results \
        = pd.read_csv(f'sharpe_results/metrics_v{variant}_w{False}_c{True}.csv', header=None).to_numpy().astype(float)[0]
      false_true_diversity_time_results \
        = pd.read_csv(f'sharpe_results/solver_times_v{variant}_w{False}_c{True}.csv', header=None).to_numpy().astype(float)[0]
      
      with open(f"collated_sharpe_results/metrics_s{setting}_a{alpha_ind}_w{False}_c{True}.csv", "at") as file:
        file.write(",".join(map(str, false_true_diversity_results)) + "\n")

      with open(f"collated_sharpe_results/solver_times_s{setting}_a{alpha_ind}_w{False}_c{True}.csv", "at") as file:
        file.write(",".join(map(str, false_true_diversity_time_results)) + "\n")
      
      true_false_diversity_results \
        = pd.read_csv(f'sharpe_results/metrics_v{variant}_w{True}_c{False}.csv', header=None).to_numpy().astype(float)[0]
      true_false_diversity_time_results \
        = pd.read_csv(f'sharpe_results/solver_times_v{variant}_w{True}_c{False}.csv', header=None).to_numpy().astype(float)[0]

      with open(f"collated_sharpe_results/metrics_s{setting}_a{alpha_ind}_w{True}_c{False}.csv", "at") as file:
        file.write(",".join(map(str, true_false_diversity_results)) + "\n")

      with open(f"collated_sharpe_results/solver_times_s{setting}_a{alpha_ind}_w{True}_c{False}.csv", "at") as file:
        file.write(",".join(map(str, true_false_diversity_time_results)) + "\n")

      
      mosek_results \
        = pd.read_csv(f'sharpe_cvxpy_results/metrics_v{variant}.csv', header=None).to_numpy().astype(float)[0]

      with open(f"collated_sharpe_results/metrics_s{setting}_a{alpha_ind}_w{False}_c{False}.csv", "at") as file:
        file.write(",".join(map(str, mosek_results)) + "\n")
      


    except:
       print(f'Result doesnt exist for variant = {variant}')
       dne += 1

for variant in tqdm(variant_indices):
    (job, setting, alpha_ind) = variants[variant]
    try:
      vanilla_results = pd.read_csv(f'sharpe_results/vanilla_metrics_v{variant}_w{True}_c{True}.csv', header=None).to_numpy().astype(float)[0]
      with open(f"collated_sharpe_results/vanilla_metrics_s{setting}_a{alpha_ind}.csv", "at") as file:
        file.write(",".join(map(str, vanilla_results)) + "\n")
    except:
       print(f'Vanilla result doesnt exist for variant = {variant}')
       vanilla_dne += 1

print(f'Dne results: {dne}')
print(f'Vanilla dne results: {vanilla_dne}')





dne = 0
vanilla_dne = 0
for variant in tqdm(variant_indices):
    (job, setting, alpha_ind) = variants[variant]
    
    try:
      histogram_results = pd.read_csv(f'optimized_sharpe_histogram_results/histogram_v{variant}.csv', header=None).to_numpy().astype(float)[0]

      with open(f"collated_sharpe_results/histogram_s{setting}_a{alpha_ind}.csv", "at") as file:
        file.write(",".join(map(str, histogram_results)) + "\n")
    except:
       print(f'Result doesnt exist for variant = {variant}')
       dne += 1

print(f'Optimized results: {dne}')