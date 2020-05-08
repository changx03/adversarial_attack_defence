#!/bin/bash
# chmod +x ./cmd/script_multi_tree.sh

# failed tests: BankNote, BreastCancerWisconsin
echo "start Iris..."
python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d Iris

echo "start BankNote..."
python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d BankNote

echo "start WheatSeed..."
python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d WheatSeed

echo "start HTRU2..."
python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d HTRU2

echo "start BreastCancer..."
python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d BreastCancerWisconsin
