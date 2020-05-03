#!/bin/bash

# chmod +x ./cmd/script_squeezing.sh

echo "start MNIST..."
python ./cmd/defend_squeeze.py -vl -e 50 -d 8 --sigma 0.2 -k 3 -m ./save/MnistCnnV2_MNIST_e50.pt -FBDCS

# echo "start CIFAR10..."
# echo "start SVHN..."
# echo "start Iris..."
# echo "start BankNote..."
# echo "start WheatSeed..."
# echo "start HTRU2..."
# echo "start BreastCancer..."
