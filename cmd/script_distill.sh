#!/bin/bash
# chmod +x ./cmd/script_distill.sh

python ./cmd/defend_distill.py -vl -t 10 -r 0.5 -m ./save/IrisNN_Iris_e200.pt -BCDF
python ./cmd/defend_distill.py -vl -t 10 -r 0.25 -m ./save/BCNN_BreastCancerWisconsin_e200.pt  -BCDF
python ./cmd/defend_distill.py -vl -t 10 -r 0.25 -m ./save/IrisNN_BankNote_e200.pt  -BCDF
python ./cmd/defend_distill.py -vl -t 10 -r 0.25 -m ./save/IrisNN_HTRU2_e200.pt  -BCDF
python ./cmd/defend_distill.py -vl -t 10 -r 0.5 -m ./save/IrisNN_WheatSeed_e300.pt  -BCDF
python ./cmd/defend_distill.py -vl -t 10 -r 0.25 -m ./save/MnistCnnV2_MNIST_e50.pt  -BCDFS
python ./cmd/defend_distill.py -vl -t 10 -r 0.25 -m ./save/CifarCnn_CIFAR10_e50.pt  -BCDFS
python ./cmd/defend_distill.py -vl -t 10 -r 0.25 -m ./save/CifarResnet50_CIFAR10_e30.pt  -BCDFS
python ./cmd/defend_distill.py -vl -t 10 -r 0.25 -m ./save/CifarResnet50_SVHN_e30.pt  -BCDFS