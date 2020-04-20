#!/bin/bash
# use chmod +x ./cmd/cv.sh to give execute permission

# python ./cmd/cross_valid.py -vl -p ./cmd/CvParams_iris.json -m ./save/IrisNN_Iris_e200.pt -a ./save/IrisNN_Iris_Carlini_adv.npy
# python ./cmd/cross_valid.py -vl -p ./cmd/CvParams_iris.json -m ./save/BCNN_BreastCancerWisconsin_e200.pt -a ./save/BCNN_BreastCancerWisconsin_Carlini_adv.npy
# python ./cmd/cross_valid.py -vl -p ./cmd/CvParams_iris.json -m ./save/IrisNN_BankNote_e200.pt -a ./save/IrisNN_BankNote_Carlini_adv.npy
# python ./cmd/cross_valid.py -vl -p ./cmd/CvParams_iris.json -m ./save/IrisNN_HTRU2_e200.pt -a ./save/IrisNN_HTRU2_Carlini_adv.npy
# python ./cmd/cross_valid.py -vl -p ./cmd/CvParams_iris.json -m ./save/IrisNN_WheatSeed_e300.pt -a ./save/IrisNN_WheatSeed_Carlini_adv.npy
python ./cmd/cross_valid.py -vl -p ./cmd/CvParams_cifar.json -m ./save/MnistCnnV2_MNIST_e50.pt -a ./save/MnistCnnV2_MNIST_Carlini_adv.npy
# python ./cmd/cross_valid.py -vl -p ./cmd/CvParams_cifar.json -m ./save/CifarCnn_CIFAR10_e50.pt -a ./save/CifarCnn_CIFAR10_Carlini_adv.npy
# python ./cmd/cross_valid.py -vl -p ./cmd/CvParams_cifar.json -m ./save/CifarResnet50_CIFAR10_e30.pt -a ./save/CifarResnet50_CIFAR10_Carlini_adv.npy
# python ./cmd/cross_valid.py -vl -p ./cmd/CvParams_cifar.json -m ./save/CifarResnet50_SVHN_e30.pt -a ./save/CifarResnet50_SVHN_Carlini_adv.npy