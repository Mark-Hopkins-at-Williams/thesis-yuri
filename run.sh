#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH --mem=10G
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --config configs/experiment1-2.bi2.1024.json
python finetune.py --config configs/experiment1-2.bi2.2048.json
python finetune.py --config configs/experiment1-2.bi2.4096.json
python finetune.py --config configs/experiment1-2.bi2.8192.json
python finetune.py --config configs/experiment1-2.bi2.16834.json
