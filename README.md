# Diversifying conformal selections

This repository contains code that can be used to reproduce the results in our paper "Diversifying conformal selections". 

## Setup

Running simulations and experiments involving the Sharpe ratio and Markowitz objective involve using the `dacs` package. Navigate to the `divSelect` directory 
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
