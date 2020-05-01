#!/bin/bash
# chmod +x ./cmd/script_attack_clean.sh

# MNIST
python ./cmd/attack.py -m ./save/MnistCnnV2_MNIST_e50.pt -p ./cmd/AttackParams.json -lvw -F
# CIFAR10
python ./cmd/attack.py -m ./save/CifarCnn_CIFAR10_e50.pt -p ./cmd/AttackParams.json -lvw -F
python ./cmd/attack.py -m ./save/CifarResnet50_CIFAR10_e50.pt -p ./cmd/AttackParams.json -lvw -F
# SVHN
python ./cmd/attack.py -m ./save/CifarCnn_SVHN_e50.pt -p ./cmd/AttackParams.json -lvw -F
python ./cmd/attack.py -m ./save/CifarResnet50_SVHN_e50.pt -p ./cmd/AttackParams.json -lvw -F
# Iris
python ./cmd/attack.py -m ./save/IrisNN_Iris_e200.pt -p ./cmd/AttackParams.json -lvw -F
# BankNote
python ./cmd/attack.py -m ./save/IrisNN_BankNote_e200.pt -p ./cmd/AttackParams.json -lvw -F
# WheatSeed
python ./cmd/attack.py -m ./save/IrisNN_WheatSeed_e300.pt -p ./cmd/AttackParams.json -lvw -F
# HTRU2
python ./cmd/attack.py -m ./save/IrisNN_HTRU2_e200.pt -p ./cmd/AttackParams.json -lvw -F
# BreastCancer
python ./cmd/attack.py -m ./save/BCNN_BreastCancerWisconsin_e200.pt -p ./cmd/AttackParams.json -lvw -F

# python ./cmd/attack.py -m ./save/CifarCnn_SVHN_e50.pt -p ./cmd/AttackParams.json -lvw -BDCS
# python ./cmd/attack.py -m ./save/CifarResnet50_SVHN_e50.pt -p ./cmd/AttackParams.json -lvw -BDCS