#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH -o faiss_sample/logs/log_%j.out
#SBATCH -e faiss_sample/logs/log_%j.err
#SBATCH --gres=gpu:1
python faiss_map.py --config config.json