#!/bin/bash
# chmod +x ./cmd/script_train.sh

python ./cmd/train.py -d BreastCancerWisconsin -e 200 -vwl
python ./cmd/train.py -d HTRU2 -e 200 -vwl
python ./cmd/train.py -d Iris -e 200 -vwl
python ./cmd/train.py -d BankNote -e 200 -vwl
python ./cmd/train.py -d WheatSeed -e 300 -vwl

python ./cmd/train.py -d MNIST -e 50 -lvw
python ./cmd/train.py -d CIFAR10 -m CifarCnn -e 50 -vwl
python ./cmd/train.py -d CIFAR10 -m CifarResnet50 -e 50 -vwl
python ./cmd/train.py -d SVHN -m CifarCnn -e 50 -vwl
python ./cmd/train.py -d SVHN -m CifarResnet50 -e 50 -vwl