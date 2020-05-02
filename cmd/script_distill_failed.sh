#!/bin/bash

# chmod +x ./cmd/script_distill_failed.sh

echo "Iris and BreastCancerWisconsin failed the initial experiment"
echo "These datasets only accept lower temperatures."

echo "Start Iris..."
python ./cmd/defend_distill.py -vl -t 10 -e 200 -m ./save/IrisNN_Iris_e200.pt -FBDC
python ./cmd/defend_distill.py -vl -t 2 -e 200 -m ./save/IrisNN_Iris_e200.pt -FBDC

echo "Start BreastCancerWisconsin..."
python ./cmd/defend_distill.py -vl -t 10 -e 200 -m ./save/BCNN_BreastCancerWisconsin_e200.pt -FBDC
python ./cmd/defend_distill.py -vl -t 2 -e 200 -m ./save/BCNN_BreastCancerWisconsin_e200.pt -FBDC