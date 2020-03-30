import argparse as ap

from aad.datasets import get_dataset_list


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', type=str, required=True, choices=get_dataset_list(),
        help='the dataset you want to train')
    parser.add_argument(
        '-o', '--ofile', type=str,
        help='the filename will be used to store model parameters')
    parser.add_argument(
        '-e', '--epoch', type=int, required=True,
        help='the number of max epochs for training')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=128, help='batch size')
    parser.add_argument(
        '-s', '--seed', type=int, default=4096,
        help='the seed for random number generator')
    parser.add_argument(
        '-H', '--shuffle', type=bool, default=True, help='shuffle the dataset')
    parser.add_argument(
        '-n', '--normalize', type=bool, default=True,
        help='apply zero mean and scaling to the dataset (for numeral dataset only)')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='set logger level to debug')
    parser.add_argument(
        '-l', '--savelog', action='store_true', help='save logging file')
    args = parser.parse_args()
    dataset = args.dataset
    filename = args.ofile
    max_epochs = args.epoch
    batch_size = args.batchsize
    seed = args.seed
    use_shuffle = args.shuffle
    use_normalize = args.normalize
    verbose = args.verbose
    save_log = args.savelog
    print('RECEIVED PARAMETERS:')
    print(f'dataset       : {dataset}')
    print(f'filename      : {filename}')
    print(f'max_epochs    : {max_epochs}')
    print(f'batch_size    : {batch_size}')
    print(f'seed          : {seed}')
    print(f'use_shuffle   : {use_shuffle}')
    print(f'use_normalize : {use_normalize}')
    print(f'verbose       : {verbose}')
    print(f'save_log      : {save_log}')


if __name__ == "__main__":
    main()
