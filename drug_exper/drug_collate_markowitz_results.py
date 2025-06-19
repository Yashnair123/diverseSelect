import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

alpha100 = int(sys.argv[1])
gammas = [0.25, 0.5, 0.75]
couple=True
dne = 0
vanilla_dne = 0

dne_indices = []

quantile_indexer = 1
for gamma_indexer in range(3):
  for job in tqdm(range(250)):
      
      try:
        diversity_results = pd.read_csv(f'markowitz_results/metrics_c{couple}_j{job}_q{quantile_indexer}_g{gamma_indexer}_a{alpha100}.csv', header=None).to_numpy().astype(float)[0]
        with open(f"collated_markowitz_results/metrics_c{couple}_q{quantile_indexer}_g{gamma_indexer}_a{alpha100}.csv", "at") as file:
          file.write(",".join(map(str, diversity_results)) + "\n")
      except:
        print(f'Result doesnt exist for couple={couple}, quantile={quantile_indexer}, job={job}')
        dne += 1

        dne_indices.append(job)

  for job in tqdm(range(250)):
      try:
        vanilla_results = pd.read_csv(f'markowitz_results/vanilla_metrics_c{couple}_j{job}_q{quantile_indexer}_g{gamma_indexer}_a{alpha100}.csv', header=None).to_numpy().astype(float)[0]
        with open(f"collated_markowitz_results/vanilla_metrics_c{couple}_q{quantile_indexer}_g{gamma_indexer}_a{alpha100}.csv", "at") as file:
          file.write(",".join(map(str, vanilla_results)) + "\n")
      except:
        print(f'Vanilla result doesnt exist for couple={couple}, quantile={quantile_indexer}, job={job}')
        vanilla_dne += 1

  print(f'Dne results: {dne}')
  print(f'Vanilla dne results: {vanilla_dne}')


  for job in (tqdm(range(250))):
    try:
      histogram_results = pd.read_csv(f'optimized_markowitz_histogram_results/histogram_c{couple}_j{job}_q{quantile_indexer}_g{gamma_indexer}_a{alpha100}.csv', header=None).to_numpy().astype(float)[0]
      with open(f"collated_markowitz_results/histogram_c{couple}_q{quantile_indexer}_g{gamma_indexer}_a{alpha100}.csv", "at") as file:
        file.write(",".join(map(str, histogram_results)) + "\n")
    except:
       print(f'optimized Result doesnt exist for variant = {job}')

