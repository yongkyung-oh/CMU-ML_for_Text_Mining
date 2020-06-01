Implemented different embedding methods
-W2V: without pretrained embedding: use the Word2Vector model generated in the preprocess
-Pre: with pretrained embedding: use the pretrained embedding data from Amazon review
-G6B: with pretrained glove 6B: use the pretrained embedding data from glove 6B: 100

python CNN.py -h
usage: CNN.py [-h] --Embedd EMBEDD [--Epochs EPOCHS]

Python module for Neural Network model for sensitive analysis

optional arguments:
  -h, --help       show this help message and exit
  --Embedd EMBEDD  Embedding methods: W2V | Pre | G6B
  --Epochs EPOCHS  Epoch for model train (default: 10)


python LSTM.py -h
usage: LSTM.py [-h] --Embedd EMBEDD [--Epochs EPOCHS]

Python module for Neural Network model for sensitive analysis

optional arguments:
  -h, --help       show this help message and exit
  --Embedd EMBEDD  Embedding methods: W2V | Pre | G6B
  --Epochs EPOCHS  Epoch for model train (default: 10)

