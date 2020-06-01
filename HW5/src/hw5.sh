#!/usr/bin/env bash

python preprocess.py CTF
python preprocess.py DF
python preprocess.py BAL

python rmlr.py CTF
python rmlr.py DF
python rmlr.py BAL


