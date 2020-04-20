#!/bin/bash
# chmod +x ./cmd/script_attack.sh

python ./cmd/attack.py -m ./save/IrisNN_Iris_e200.pt -p ./cmd/AttackParams.json -lvw -BCDF
python ./cmd/attack.py -m ./save/BCNN_BreastCancerWisconsin_e200.pt -p ./cmd/AttackParams.json -lvw -BCDF
python ./cmd/attack.py -m ./save/IrisNN_BankNote_e200.pt -p ./cmd/AttackParams.json -lvw -BCDF
python ./cmd/attack.py -m ./save/IrisNN_HTRU2_e200.pt -p ./cmd/AttackParams.json -lvw -BCDF
python ./cmd/attack.py -m ./save/IrisNN_WheatSeed_e300.pt -p ./cmd/AttackParams.json -lvw -BCDF
python ./cmd/attack.py -m ./save/MnistCnnV2_MNIST_e50.pt -p ./cmd/AttackParams.json -lvw -BCDFS
python ./cmd/attack.py -m ./save/CifarCnn_CIFAR10_e50.pt -p ./cmd/AttackParams.json -lvw -BCDFS
python ./cmd/attack.py -m ./save/CifarResnet50_CIFAR10_e30.pt -p ./cmd/AttackParams.json -lvw -BCDFS
python ./cmd/attack.py -m ./save/CifarResnet50_SVHN_e30.pt -p ./cmd/AttackParams.json -lvw -BCDFS