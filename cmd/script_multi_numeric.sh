#!/bin/bash
# chmod +x ./cmd/script_multi_numeric.sh

echo "start Iris..."
python ./cmd/multi_numeric.py -l -i 100 -e 200 -d Iris

echo "start BankNote..."
python ./cmd/multi_numeric.py -l -i 100 -e 200 -d BankNote

echo "start WheatSeed..."
python ./cmd/multi_numeric.py -l -i 100 -e 300 -d WheatSeed

echo "start HTRU2..."
python ./cmd/multi_numeric.py -l -i 100 -e 200 -d HTRU2

echo "start BreastCancer..."
python ./cmd/multi_numeric.py -l -i 100 -e 200 -d BreastCancerWisconsin