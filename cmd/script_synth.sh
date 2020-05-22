#!/bin/bash
# chmod +x ./cmd/script_synth.sh

# Experiment 1:
python ./cmd/synth_sample.py -vl -s 250 -f 30 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 500 -f 30 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 1000 -f 30 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 5000 -f 30 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 10000 -f 30 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 50000 -f 30 -c 2 -i 1 -e 200

# Experiment 2:
python ./cmd/synth_sample.py -vl -s 5000 -f 4 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 5000 -f 8 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 5000 -f 16 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 5000 -f 32 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 5000 -f 64 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 5000 -f 128 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 5000 -f 256 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 5000 -f 512 -c 2 -i 1 -e 200

# Experiment 3:
python ./cmd/synth_sample.py -vl -s 500 -f 25 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 1000 -f 50 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 10000 -f 500 -c 2 -i 1 -e 200

python ./cmd/synth_sample.py -vl -s 500 -f 12 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 1000 -f 25 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 10000 -f 250 -c 2 -i 1 -e 200

python ./cmd/synth_sample.py -vl -s 500 -f 6 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 1000 -f 12 -c 2 -i 1 -e 200
python ./cmd/synth_sample.py -vl -s 10000 -f 120 -c 2 -i 1 -e 200
