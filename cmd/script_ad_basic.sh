#!/bin/bash

# chmod +x ./cmd/script_ad_basic.sh

# CIFAR10 and SVHN both have 2 seperate model avaliable, `basic` and `ResNet50`.
# This script trains the basic model.

# Appliciability Domain on CIFAR10 basic model
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_BIM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_Carlini_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_DeepFool_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_FGSM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_Saliency_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt

# Appliciability Domain on SVHN basic model
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_SVHN_BIM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_SVHN_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_SVHN_Carlini_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_SVHN_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_SVHN_DeepFool_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_SVHN_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_SVHN_FGSM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_SVHN_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_SVHN_Saliency_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_SVHN_e50.pt