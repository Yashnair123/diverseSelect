# Diversifying conformal selections

This repository contains code that can be used to reproduce the results in our paper "Diversifying conformal selections". 

## Setup

Running simulations and experiments involving the Sharpe ratio and Markowitz objective involve using our `dacs` package, which contains custom projected gradient descent (PGD) solvers. Navigate to the `divSelect` directory 
and 

* Install `dacs`:
    ```bash
    pip install -e . --verbose
    ```
    
* The install prerequisites are:
    ```bash
    mamba update -y conda mamba
    mamba env create -f environment.yml
    mamba activate dacs
    poetry install --no-root
    ```
## Usage

### Simulations
`sharpe-markowitz_sims_settings1-2` and `sharpe-markowitz_sims_settings3-4` contain code to reproduce all of our simulation results involving the Sharpe ratio and Markowitz objective. The files `sherlock_sharpe_driver.py` and `sherlock_markowitz_driver.py` run our method, using our custom PGD solvers. The scripts `sherlock_sharpe_driver_cvxpy.py` or `sherlock_markowitz_driver_cvxpy.py` in `sharpe-markowitz_sims_settings1-2` reproduce results using the MOSEK solver for reward computation. Additionally, the scripts `optimized_sharpe_get_histogram.py` and `optimized_markowitz_get_histogram.py` are used to construct the baseline distributions $\widehat{F}^{\varphi}_{\mathrm{baseline}}$ described in the paper.

The directory `underrep_sims` contains code for our underrepresentation index simulations. To reproduce our results, run either `sherlock_cluster_driver.py` or `sherlock_cluster_driver_100.py` (the latter contains results for $m=100$ test samples).

### Experiments
`drug_exper` contains code for our drug discovery experiments. The file `data_preprocess.ipynb` cleans the data, constructs the Tanimoto similarity matrix, and generates random train/calibration/test splits of the dataset. The file `sherlock_train_dti.py` constructs calibration and test scores, corresponding to each of these train-test splits, by training a small neural neural network on the training data. The files `sherlock_sharpe_drug_driver.py` and `sherlock_markowitz_drug_driver.py` run our method, using the custom PGD solvers, and `drug_optimized_sharpe_results.py` and `drug_optimized_markowitz_results.py` construct the baseline distributions $\widehat{F}^{\varphi}_{\mathrm{baseline}}$. The raw original dataset for these experiments is from: https://www.guidetopharmacology.org/DATA/gpcr_interactions.csv.


`hiring_exper` contains code for our job hiring experiments. All preprocessing code as well as code to run our method is found in the notebook `hiring_exper.ipynb`. The dataset for these experiments is from https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement.



