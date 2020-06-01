#!/bin/bash
python3 pagerank_sample.py

# run pagerank
python3 pagerank.py --Method GPR
python3 pagerank.py --Method QTSPR
python3 pagerank.py --Method PTSPR

# run information retrieval
python3 retrieval.py --Method GPR --Score NS
python3 retrieval.py --Method GPR --Score WS
python3 retrieval.py --Method GPR --Score CM

python3 retrieval.py --Method QTSPR --Score NS
python3 retrieval.py --Method QTSPR --Score WS
python3 retrieval.py --Method QTSPR --Score CM

python3 retrieval.py --Method PTSPR --Score NS
python3 retrieval.py --Method PTSPR --Score WS
python3 retrieval.py --Method PTSPR --Score CM


# run pagerank and save
python3 pagerank.py --Method GPR --Save True
python3 pagerank.py --Method QTSPR --Save True
python3 pagerank.py --Method PTSPR --Save True

# run information retrieval using data import
python3 retrieval.py --Method GPR --Score NS
python3 retrieval.py --Method GPR --Score WS
python3 retrieval.py --Method GPR --Score CM

python3 retrieval.py --Method QTSPR --Score NS
python3 retrieval.py --Method QTSPR --Score WS
python3 retrieval.py --Method QTSPR --Score CM

python3 retrieval.py --Method PTSPR --Score NS
python3 retrieval.py --Method PTSPR --Score WS
python3 retrieval.py --Method PTSPR --Score CM

