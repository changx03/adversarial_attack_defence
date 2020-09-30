import os
import codecs
from setuptools import setup, find_packages

install_requires = [
    'adversarial-robustness-toolbox==1.4.0',
    'scikit-learn==0.22.2',
    'numpy>=1.15.4',
    'pillow>=6.2.0',
    'python-dateutil>=2.7.3',
    'astroid',
    'jupyterlab',
    'launchpadlib',
    'matplotlib',
    'pandas',
    'torch',
    'torchvision',
]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(name='aad', version=get_version(
    "aad/__init__.py"), 
    packages=find_packages(), 
    install_requires=install_requires)
