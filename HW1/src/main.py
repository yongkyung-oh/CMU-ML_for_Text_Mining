#!/bin/bash

import os
os.system('python pagerank_sample.py')

# run pagerank
os.system('python3 pagerank.py --Method GPR')
os.system('python3 pagerank.py --Method QTSPR')
os.system('python3 pagerank.py --Method PTSPR')

# run information retrieval
os.system('python3 retrieval.py --Method GPR --Score NS')
os.system('python3 retrieval.py --Method GPR --Score WS')
os.system('python3 retrieval.py --Method GPR --Score CM')

os.system('python3 retrieval.py --Method QTSPR --Score NS')
os.system('python3 retrieval.py --Method QTSPR --Score WS')
os.system('python3 retrieval.py --Method QTSPR --Score CM')

os.system('python3 retrieval.py --Method PTSPR --Score NS')
os.system('python3 retrieval.py --Method PTSPR --Score WS')
os.system('python3 retrieval.py --Method PTSPR --Score CM')
