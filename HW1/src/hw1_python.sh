#!/bin/bash
python pagerank_sample.py

# run pagerank
python pagerank.py --Method GPR
python pagerank.py --Method QTSPR
python pagerank.py --Method PTSPR

# run information retrieval
python retrieval.py --Method GPR --Score NS
python retrieval.py --Method GPR --Score WS
python retrieval.py --Method GPR --Score CM

python retrieval.py --Method QTSPR --Score NS
python retrieval.py --Method QTSPR --Score WS
python retrieval.py --Method QTSPR --Score CM

python retrieval.py --Method PTSPR --Score NS
python retrieval.py --Method PTSPR --Score WS
python retrieval.py --Method PTSPR --Score CM


# run pagerank and save
python pagerank.py --Method GPR --Save True
python pagerank.py --Method QTSPR --Save True
python pagerank.py --Method PTSPR --Save True

# run information retrieval using data import
python retrieval.py --Method GPR --Score NS
python retrieval.py --Method GPR --Score WS
python retrieval.py --Method GPR --Score CM

python retrieval.py --Method QTSPR --Score NS
python retrieval.py --Method QTSPR --Score WS
python retrieval.py --Method QTSPR --Score CM

python retrieval.py --Method PTSPR --Score NS
python retrieval.py --Method PTSPR --Score WS
python retrieval.py --Method PTSPR --Score CM

