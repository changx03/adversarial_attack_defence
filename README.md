# Testing Adversarial Attacks and Defences

## Build and run the package

Install module as a package using `setuptools`

1. Use a virtual environment

    ```bash
    virtualenv --system-site-packages -p python3 ./venv
    source ./venv/bin/activate  # sh, bash, ksh, or zsh
    ```

1. Install you project from `pip`

    ```bash
    pip install -e .
    pip freeze
    ```

1. Add `myPackageDemo.` into your imports

    ```python
    from myPackageDemo.api.api import function_from_api
