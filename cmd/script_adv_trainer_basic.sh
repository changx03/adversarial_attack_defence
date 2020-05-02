#!/bin/bash

# chmod +x ./cmd/script_adv_trainer_basic.sh

# CIFAR10 and SVHN both have 2 seperate model avaliable, `basic` and `ResNet50`.
# This script trains the basic model.

echo "CIFAR10 on Basic model"
python ./cmd/defend_advtr.py -vlt -e 30 -r 0.25 -m ./save/CifarCnn_CIFAR10_e50.pt -BCDFS

echo "SVHN on Basic model"
python ./cmd/defend_advtr.py -vlt -e 30 -r 0.25 -m ./save/CifarCnn_SVHN_e50.pt -BCDFS

echo "CIFAR10 on Resnet50"
python ./cmd/defend_advtr.py -vlt -e 30 -r 0.25 -m ./save/CifarResnet50_CIFAR10_e50.pt -BCDFS

echo "SVHN on Resnet50"
python ./cmd/defend_advtr.py -vlt -e 30 -r 0.25 -m ./save/CifarResnet50_SVHN_e50.pt -BCDFS