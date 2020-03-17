import logging
import os

from aad import attacks, basemodels, datasets, defences

# Semantic Version
__version__ = '0.0.1'

if not os.path.exists('log'):
    os.makedirs('log')

logging.basicConfig(
    # filename=os.path.join('log', 'aad.log'),
    format='%(asctime)s:%(levelname)s:%(module)s:%(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
