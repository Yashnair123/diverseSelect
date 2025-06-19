import numpy as np
import pandas as pd
from tqdm import tqdm
import sys



def is_csv_empty(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        return not content.strip()



given_setting = int(sys.argv[2])
couple = bool(int(sys.argv[1]))
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
      diversity_results = pd.read_csv(f'sharpe_results/metrics_v{variant}.csv', header=None).to_numpy().astype(float)[0]
      with open(f"collated_sharpe_results/metrics_c{couple}_s{setting}_a{alpha_ind}.csv", "at") as file:
        file.write(",".join(map(str, diversity_results)) + "\n")
    except:
       print(f'Result doesnt exist for variant = {variant}')
       dne += 1

for variant in tqdm(variant_indices):
    (job, setting, alpha_ind) = variants[variant]
    try:
      vanilla_results = pd.read_csv(f'sharpe_results/vanilla_metrics_v{variant}.csv', header=None).to_numpy().astype(float)[0]
      with open(f"collated_sharpe_results/vanilla_metrics_c{couple}_s{setting}_a{alpha_ind}.csv", "at") as file:
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
      if is_csv_empty(f'optimized_sharpe_histogram_results/histogram_v{variant}.csv'):
         histogram_results = [0.]
      else:
        histogram_results = pd.read_csv(f'optimized_sharpe_histogram_results/histogram_v{variant}.csv', header=None).to_numpy().astype(float)[0]

      with open(f"collated_sharpe_results/histogram_c{couple}_s{setting}_a{alpha_ind}.csv", "at") as file:
        file.write(",".join(map(str, histogram_results)) + "\n")
    except:
       print(f'Result doesnt exist for variant = {variant}')
       dne += 1

print(f'Optimized results: {dne}')