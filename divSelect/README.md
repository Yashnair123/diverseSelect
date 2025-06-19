# divSelect

## Installation

* Install prerequisites:
    ```bash
    mamba update -y conda mamba
    mamba env create -f environment.yml
    mamba activate dacs
    poetry install --no-root
    ```

* Install `dacs`:
    ```bash
    pip install -e . --verbose
    ```