# CERWU
Repository for the paper _Reducing Storage of Pretrained Neural Networks by Rate-Constrained Quantization and Entropy Coding_, 
available under _[Link will be added upon publication]_.

## Installation
Before starting, make sure you have a C++17-compatible compiler installed. We also recommend using Python 3.12, preferrably in a fresh venv using an environment of your choice. 
If you have downloaded this project from git, make sure that the submodules are updated:
```git submodule update```
Afterwards, install the project via executing `make install` in the root directory.

## Usage
To test your installation, you can run tests via 

```pytest tests```. 

Make sure you have pytest installed (either via installing the dev dependencies below or by running ```pip install pytest```).

To perform an experiment, use the script `experiment_runner.py`. Example usage (run in the project root):

```python3 scripts/experiment_runner.py network=resnet18_cifar10 compression=cerwu compression.log10_lm=-8 evaluation=cifar10_small calibration=cifar10_small device=cpu ```

If this is your first time running the algorithm, it will take some time to calculate the hessians. A demo can also be seen in `demo.ipynb`. 

## Developer information
Follow the installation instructions from above. Afterwards, run

```pip install ".[dev]"```

in the project root.

### Git hooks
Before commiting, install the githooks via 

```python3 -m python_githooks```

## Authors
- Alexander Conzelmann
- Robert Bamler
