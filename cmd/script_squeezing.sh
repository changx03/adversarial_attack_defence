#!/bin/bash
# chmod +x ./cmd/script_squeezing.sh

# echo "start MNIST..."
# python ./cmd/defend_squeeze.py -vl -e 50 -d 8 --sigma 0.2 -k 3 -m ./save/MnistCnnV2_MNIST_e50.pt -FBDCS

# echo "start CIFAR10..."
# python ./cmd/defend_squeeze.py -vl -e 50 -d 8 --sigma 0.2 -k 3 -m ./save/CifarCnn_CIFAR10_e50.pt -FBDCS
# python ./cmd/defend_squeeze.py -vl -e 50 -d 8 --sigma 0.2 -k 3 -m ./save/CifarResnet50_CIFAR10_e50.pt -FBDCS

# echo "start SVHN..."
# python ./cmd/defend_squeeze.py -vl -e 50 -d 8 --sigma 0.2 -k 3 -m ./save/CifarCnn_SVHN_e50.pt -FBDCS
# python ./cmd/defend_squeeze.py -vl -e 50 -d 8 --sigma 0.2 -k 3 -m ./save/CifarResnet50_SVHN_e50.pt -FBDCS

# echo "start Iris..."
# python ./cmd/defend_squeeze.py -vl -e 200 -d 8 --sigma 0.2 -m ./save/IrisNN_Iris_e200.pt -FBDC

echo "start BankNote..."
python ./cmd/defend_squeeze.py -vl -e 200 -d 8 --sigma 0.2 -m ./save/IrisNN_BankNote_e200.pt -FBDC

echo "start WheatSeed..."
python ./cmd/defend_squeeze.py -vl -e 300 -d 8 --sigma 0.2 -m ./save/IrisNN_WheatSeed_e300.pt -FBDC

echo "start HTRU2..."
python ./cmd/defend_squeeze.py -vl -e 200 -d 8 --sigma 0.2 -m ./save/IrisNN_HTRU2_e200.pt -FBDC

# echo "start BreastCancer..."
# python ./cmd/defend_squeeze.py -vl -e 200 -d 8 --sigma 0.2 -m ./save/BCNN_BreastCancerWisconsin_e200.pt -FBDC

# Datasets do NOT work in the initial run
#   - BankNote
#   - HTRU2
#   - WheatSeed