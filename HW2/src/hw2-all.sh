#!/usr/bin/env bash
python main.py --Method user --File dev --Similarity dot --Mean mean --K 10
python main.py --Method user --File dev --Similarity dot --Mean mean --K 100
python main.py --Method user --File dev --Similarity dot --Mean mean --K 500

python main.py --Method user --File dev --Similarity cosine --Mean mean --K 10
python main.py --Method user --File dev --Similarity cosine --Mean mean --K 100
python main.py --Method user --File dev --Similarity cosine --Mean mean --K 500

python main.py --Method user --File dev --Similarity cosine --Mean weight --K 10
python main.py --Method user --File dev --Similarity cosine --Mean weight --K 100
python main.py --Method user --File dev --Similarity cosine --Mean weight --K 500


python main.py --Method item --File dev --Similarity dot --Mean mean --K 10
python main.py --Method item --File dev --Similarity dot --Mean mean --K 100
python main.py --Method item --File dev --Similarity dot --Mean mean --K 500

python main.py --Method item --File dev --Similarity cosine --Mean mean --K 10
python main.py --Method item --File dev --Similarity cosine --Mean mean --K 100
python main.py --Method item --File dev --Similarity cosine --Mean mean --K 500

python main.py --Method item --File dev --Similarity cosine --Mean weight --K 10
python main.py --Method item --File dev --Similarity cosine --Mean weight --K 100
python main.py --Method item --File dev --Similarity cosine --Mean weight --K 500


python main.py --Method pcc_user --File dev --Similarity dot --Mean mean --K 10
python main.py --Method pcc_user --File dev --Similarity dot --Mean mean --K 100
python main.py --Method pcc_user --File dev --Similarity dot --Mean mean --K 500

python main.py --Method pcc_user --File dev --Similarity cosine --Mean mean --K 10
python main.py --Method pcc_user --File dev --Similarity cosine --Mean mean --K 100
python main.py --Method pcc_user --File dev --Similarity cosine --Mean mean --K 500

python main.py --Method pcc_user --File dev --Similarity cosine --Mean weight --K 10
python main.py --Method pcc_user --File dev --Similarity cosine --Mean weight --K 100
python main.py --Method pcc_user --File dev --Similarity cosine --Mean weight --K 500



python main.py --Method pcc_item --File dev --Similarity dot --Mean mean --K 10
python main.py --Method pcc_item --File dev --Similarity dot --Mean mean --K 100
python main.py --Method pcc_item --File dev --Similarity dot --Mean mean --K 500

python main.py --Method pcc_item --File dev --Similarity cosine --Mean mean --K 10
python main.py --Method pcc_item --File dev --Similarity cosine --Mean mean --K 100
python main.py --Method pcc_item --File dev --Similarity cosine --Mean mean --K 500

python main.py --Method pcc_item --File dev --Similarity cosine --Mean weight --K 10
python main.py --Method pcc_item --File dev --Similarity cosine --Mean weight --K 100
python main.py --Method pcc_item --File dev --Similarity cosine --Mean weight --K 500


python main.py --Method pmf_gd --File dev --Latent 2 --Lambda 0.2
python main.py --Method pmf_gd --File dev --Latent 5 --Lambda 0.2
python main.py --Method pmf_gd --File dev --Latent 10 --Lambda 0.2
python main.py --Method pmf_gd --File dev --Latent 20 --Lambda 0.2
python main.py --Method pmf_gd --File dev --Latent 50 --Lambda 0.2

python main.py --Method pmf_torch --File dev --Latent 2 --Lambda 0.2
python main.py --Method pmf_torch --File dev --Latent 5 --Lambda 0.2
python main.py --Method pmf_torch --File dev --Latent 10 --Lambda 0.2
python main.py --Method pmf_torch --File dev --Latent 20 --Lambda 0.2
python main.py --Method pmf_torch --File dev --Latent 50 --Lambda 0.2


