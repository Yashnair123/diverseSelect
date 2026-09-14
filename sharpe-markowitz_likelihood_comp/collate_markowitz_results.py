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
for gamma_indexer in range(3):
  for job in range(250):
    for use_likelihoods in [0,1,2]:
      for alpha_ind in range(3):
        variants.append((job, use_likelihoods, gamma_indexer, alpha_ind))
        variant_indices.append(variant_counter)
        variant_counter += 1
# for gamma_indexer in range(3):
#   for job in range(250):
#     for setting_ind in range(2):
#       for alpha_ind in range(3):
#         variants.append((job, setting_ind, gamma_indexer, alpha_ind))
#         if setting_ind == given_setting:
#             variant_indices.append(variant_counter)
#         variant_counter += 1
dne = 0
vanilla_dne = 0
for variant in tqdm(variant_indices):
    (job, use_likelihoods, gamma_indexer, alpha_ind) = variants[variant]
    try:
      diversity_results = pd.read_csv(f'markowitz_results/metrics_v{variant}.csv', header=None).to_numpy().astype(float)[0]
      with open(f"collated_markowitz_results/metrics_c{couple}_u{use_likelihoods}_a{alpha_ind}_g{gamma_indexer}.csv", "at") as file:
        file.write(",".join(map(str, diversity_results)) + "\n")
    except:
       print(f'Result doesnt exist for variant = {variant}')
       dne += 1

for variant in tqdm(variant_indices):
    (job, use_likelihoods, gamma_indexer, alpha_ind) = variants[variant]
    try:
      vanilla_results = pd.read_csv(f'markowitz_results/vanilla_metrics_v{variant}.csv', header=None).to_numpy().astype(float)[0]
      with open(f"collated_markowitz_results/vanilla_metrics_c{couple}_u{use_likelihoods}_a{alpha_ind}_g{gamma_indexer}.csv", "at") as file:
        file.write(",".join(map(str, vanilla_results)) + "\n")
    except:
       print(f'Vanilla result doesnt exist for variant = {variant}')
       vanilla_dne += 1

print(f'Dne results: {dne}')
print(f'Vanilla dne results: {vanilla_dne}')


variants = []
variant_indices = []
variant_counter = 0
for gamma_indexer in range(3):
  for job in range(250):
      for alpha_ind in range(3):
        variants.append((job, gamma_indexer, alpha_ind))
        variant_indices.append(variant_counter)
        variant_counter += 1

dne = 0
vanilla_dne = 0
for variant in tqdm(variant_indices):
    (job, gamma_indexer, alpha_ind) = variants[variant]
    try:
      if is_csv_empty(f'optimized_markowitz_histogram_results/histogram_v{variant}.csv'):
         histogram_results = [0.]
      else:
        histogram_results = pd.read_csv(f'optimized_markowitz_histogram_results/histogram_v{variant}.csv', header=None).to_numpy().astype(float)[0]
      with open(f"collated_markowitz_results/histogram_c{couple}_a{alpha_ind}_g{gamma_indexer}.csv", "at") as file:
        file.write(",".join(map(str, histogram_results)) + "\n")
    except:
       print(f'Result doesnt exist for variant = {variant}')
       dne += 1

print(f'Optimized results: {dne}')