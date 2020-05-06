#!/bin/bash
# chmod +x ./cmd/script_tree.sh

echo "start Iris..."
python ./cmd/defend_tree.py -vl -p ./cmd/AdParamsNumeral.json -d Iris -BCDF

echo "start BankNote..."
python ./cmd/defend_tree.py -vl -p ./cmd/AdParamsNumeral.json -d BankNote -BCDF

echo "start WheatSeed..."
python ./cmd/defend_tree.py -vl -p ./cmd/AdParamsNumeral.json -d WheatSeed -BCDF

echo "start HTRU2..."
python ./cmd/defend_tree.py -vl -p ./cmd/AdParamsNumeral.json -d HTRU2 -BCDF

echo "start BreastCancer..."
python ./cmd/defend_tree.py -vl -p ./cmd/AdParamsNumeral.json -d BreastCancerWisconsin -BCDF

