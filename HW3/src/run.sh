#!/usr/bin/env bash

python preprocess.py
python CNN.py --Embedd W2V --Epochs 20
python CNN.py --Embedd Pre --Epochs 20
python CNN.py --Embedd G6B --Epochs 20

python LSTM.py --Embedd W2V --Epochs 20
python LSTM.py --Embedd Pre --Epochs 20
python LSTM.py --Embedd G6B --Epochs 20


