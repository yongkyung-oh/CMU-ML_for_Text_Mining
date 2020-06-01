#!/bin/bash
# Sample script file to run your code. Feel free to change it.
# Run this script using ./hw4.sh train_file test_file
# Example:  ./hw4.sh ../data/covtype.scale.trn.libsvm ../data/covtype.scale.tst.libsvm


echo "Running using train file at" $1 "and test file at" $2
python3 main.py $1 $2
