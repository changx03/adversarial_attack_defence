#!/bin/bash
# chmod +x ./cmd/script_squeezing_tree.sh

echo "start Iris..."
python ./cmd/multi_squeeze_tree.py -vl -i 100 -d Iris --depth 8 -s 0.2

echo "start BankNote..."
python ./cmd/multi_squeeze_tree.py -vl -i 100 -d BankNote --depth 8 -s 0.2

echo "start WheatSeed..."
python ./cmd/multi_squeeze_tree.py -vl -i 100 -d WheatSeed --depth 8 -s 0.2

echo "start HTRU2..."
python ./cmd/multi_squeeze_tree.py -vl -i 100 -d HTRU2 --depth 8 -s 0.2

echo "start BreastCancer..."
python ./cmd/multi_squeeze_tree.py -vl -i 100 -d BreastCancerWisconsin --depth 8 -s 0.2
