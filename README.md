# Adversarial Attacks and Defences

Adversarial Attacks and Defences (AAD) is a Python framework for defending machine learning models from adversarial examples.

## Required Libraries

- PyTorch (www.pytorch.org)
- ART (https://github.com/IBM/adversarial-robustness-toolbox)

## Build And Install The Package

Install module as a package using `setuptools`

1. Use a virtual environment

   ```bash
   virtualenv --system-site-packages -p python3 ./venv
   source ./venv/bin/activate
   ```

1. Install you project from `pip`

   ```bash
   pip install --upgrade pip
   pip install -e .
   pip check
   pip freeze
   ```

1. Run the code demo from Jupyter Lab

   ```bash
   cd ./examples
   jupyter lab
   ```

1. Run the script from terminal

   ```bash
   python ./cmd/train.py -d MNIST -e 5 -vw
   ```

## Code Structure

```bash
root
├─┬ aad
│ ├── attacks    # modules of adversarial attacks
│ ├── basemodels # modules of base classification models
│ ├── datasets   # data loader helper module
│ └── defences   # modules of adversarial defences
├── cmd          # scripts for terminal interface
├── data         # dataset
├── examples     # code examples
├── log          # logging files
├── save         # saved pre-trained models
├── tests        # unit tests
```

## Run Script From Terminal

The terminal scripts are separated into 3 parts: **train**, **attack** and **defence**.

- To train a model:

  ```bash
  python ./cmd/train.py --help
  ```

- To attack a model:

  ```bash
  python ./cmd/attack.py --help
  ```

- To defend a model:

  ```bash
  python ./cmd/defend_ad.py --help
  ```

## Examples

Examples are available under the `./examples/` folder.
