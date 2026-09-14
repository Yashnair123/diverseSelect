import numpy as np
import pandas as pd
from tqdm import tqdm
import sys



def is_csv_empty(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        return not content.strip()



couple = bool(int(sys.argv[1]))
variant_indices = []
variants = []
variant_counter = 0
for job in range(250):
    for use_likelihoods in [0,1,2]: # 0 means no likelihoods (exch), 1 means estimated likelihoods, 2 means true likelihoods
        for alpha_ind in range(3):
            variants.append((job, use_likelihoods, alpha_ind))
            variant_indices.append(variant_counter)
            variant_counter += 1
# for job in range(250):
#   for setting_ind in range(2):
#     for alpha_ind in range(3):
#       variants.append((job, setting_ind, alpha_ind))
#       if setting_ind == given_setting:
#         variant_indices.append(variant_counter)
#       variant_counter += 1
dne = 0
vanilla_dne = 0
for variant in tqdm(variant_indices):
    (job, use_likelihoods, alpha_ind) = variants[variant]
    try:
      diversity_results = pd.read_csv(f'sharpe_results/metrics_v{variant}.csv', header=None).to_numpy().astype(float)[0]
      with open(f"collated_sharpe_results/metrics_c{couple}_u{use_likelihoods}_a{alpha_ind}.csv", "at") as file:
        file.write(",".join(map(str, diversity_results)) + "\n")
    except:
       print(f'Result doesnt exist for variant = {variant}')
       dne += 1

for variant in tqdm(variant_indices):
    (job, use_likelihoods, alpha_ind) = variants[variant]
    try:
      vanilla_results = pd.read_csv(f'sharpe_results/vanilla_metrics_v{variant}.csv', header=None).to_numpy().astype(float)[0]
      with open(f"collated_sharpe_results/vanilla_metrics_c{couple}_u{use_likelihoods}_a{alpha_ind}.csv", "at") as file:
        file.write(",".join(map(str, vanilla_results)) + "\n")
    except:
       print(f'Vanilla result doesnt exist for variant = {variant}')
       vanilla_dne += 1

print(f'Dne results: {dne}')
print(f'Vanilla dne results: {vanilla_dne}')




dne = 0
vanilla_dne = 0

variants = []
variant_indices = []
variant_counter = 0
for job in range(250):
    for alpha_ind in range(3):
      variants.append((job, alpha_ind))
      variant_indices.append(variant_counter)
      variant_counter += 1



for variant in tqdm(variant_indices):
    (job, alpha_ind) = variants[variant]
    
    
    try:
      if is_csv_empty(f'optimized_sharpe_histogram_results/histogram_v{variant}.csv'):
         histogram_results = [0.]
      else:
        histogram_results = pd.read_csv(f'optimized_sharpe_histogram_results/histogram_v{variant}.csv', header=None).to_numpy().astype(float)[0]

      with open(f"collated_sharpe_results/histogram_c{couple}_a{alpha_ind}.csv", "at") as file:
        file.write(",".join(map(str, histogram_results)) + "\n")
    except:
       print(f'Result doesnt exist for variant = {variant}')
       dne += 1

print(f'Optimized results: {dne}')