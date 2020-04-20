#!/bin/bash
# chmod +x ./cmd/script_ad.sh

python ./cmd/defend_ad.py -vl -a ./save/BCNN_BreastCancerWisconsin_BIM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/BCNN_BreastCancerWisconsin_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/BCNN_BreastCancerWisconsin_Carlini_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/BCNN_BreastCancerWisconsin_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/BCNN_BreastCancerWisconsin_DeepFool_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/BCNN_BreastCancerWisconsin_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/BCNN_BreastCancerWisconsin_FGSM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/BCNN_BreastCancerWisconsin_e200.pt

python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_BIM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_Carlini_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_DeepFool_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_FGSM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarCnn_CIFAR10_Saliency_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarCnn_CIFAR10_e50.pt

python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_CIFAR10_BIM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarResnet50_CIFAR10_e30.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_CIFAR10_Carlini_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarResnet50_CIFAR10_e30.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_CIFAR10_DeepFool_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarResnet50_CIFAR10_e30.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_CIFAR10_FGSM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarResnet50_CIFAR10_e30.pt
python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_CIFAR10_Saliency_adv.npy -p ./cmd/AdParamsImage.json -m ./save/CifarResnet50_CIFAR10_e30.pt

python ./cmd/defend_ad.py -vl -a ./save/IrisNN_BankNote_BIM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_BankNote_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_BankNote_Carlini_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_BankNote_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_BankNote_DeepFool_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_BankNote_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_BankNote_FGSM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_BankNote_e200.pt

python ./cmd/defend_ad.py -vl -a ./save/IrisNN_HTRU2_BIM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_HTRU2_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_HTRU2_Carlini_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_HTRU2_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_HTRU2_DeepFool_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_HTRU2_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_HTRU2_FGSM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_HTRU2_e200.pt

python ./cmd/defend_ad.py -vl -a ./save/IrisNN_Iris_BIM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_Iris_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_Iris_Carlini_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_Iris_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_Iris_DeepFool_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_Iris_e200.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_Iris_FGSM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_Iris_e200.pt

python ./cmd/defend_ad.py -vl -a ./save/IrisNN_WheatSeed_BIM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_WheatSeed_e300.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_WheatSeed_Carlini_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_WheatSeed_e300.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_WheatSeed_DeepFool_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_WheatSeed_e300.pt
python ./cmd/defend_ad.py -vl -a ./save/IrisNN_WheatSeed_FGSM_adv.npy -p ./cmd/AdParamsNumeral.json -m ./save/IrisNN_WheatSeed_e300.pt

python ./cmd/defend_ad.py -vl -a ./save/MnistCnnV2_MNIST_BIM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/MnistCnnV2_MNIST_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/MnistCnnV2_MNIST_Carlini_adv.npy -p ./cmd/AdParamsImage.json -m ./save/MnistCnnV2_MNIST_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/MnistCnnV2_MNIST_DeepFool_adv.npy -p ./cmd/AdParamsImage.json -m ./save/MnistCnnV2_MNIST_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/MnistCnnV2_MNIST_FGSM_adv.npy -p ./cmd/AdParamsImage.json -m ./save/MnistCnnV2_MNIST_e50.pt
python ./cmd/defend_ad.py -vl -a ./save/MnistCnnV2_MNIST_Saliency_adv.npy -p ./cmd/AdParamsImage.json -m ./save/MnistCnnV2_MNIST_e50.pt
