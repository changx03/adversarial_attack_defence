import argparse as ap
import logging
import os

import aad.attacks as attacks

logger = logging.getLogger('attack')


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='pretrained model file name. The filename should in "<model>_<dataset>_e<max epochs>[_<date>].pt" format')
    parser.add_argument(
        '-n', '--number', type=int, default=100,
        help='the number of adversarial examples want to generate. (if more than test set, it uses all test examples.)')
    parser.add_argument(
        '-s', '--seed', type=int, default=4096,
        help='the seed for random number generator')
    args = parser.parse_args()


if __name__ == '__main__':
    main()
