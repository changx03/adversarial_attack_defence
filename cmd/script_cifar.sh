#!/bin/bash

python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_DeepFool_adv.npy -p ./cmd/AdParamsLarge.json -m ./save/CifarResnet50_SVHN_e30.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_DeepFool_adv.npy -p ./cmd/AdParamsLargeNoS2.json -m ./save/CifarResnet50_SVHN_e30.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_FGSM_adv.npy -p ./cmd/AdParamsLarge.json -m ./save/CifarResnet50_SVHN_e30.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_FGSM_adv.npy -p ./cmd/AdParamsLargeNoS2.json -m ./save/CifarResnet50_SVHN_e30.pt