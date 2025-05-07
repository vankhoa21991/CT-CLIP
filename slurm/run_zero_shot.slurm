#!/bin/bash
##File automaticaly generated on Mon Jun 14 16:26:53 2021
#SBATCH --job-name=tr132F0
#SBATCH --time=1000:00:00
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000
#SBATCH --partition=median
#SBATCH --mail-user=van-khoa.le@mediantechnologies.com
#SBATCH --mail-type=ALL
#
#SBATCH --comment=trainEx8
#SBATCH --chdir=/mnt/home/admvkl@median.cad/slurm_log/
#SBATCH --verbose

nvidia-smi

eval "$(conda shell.bash hook)"
conda activate venv_py39

cd /mnt/home/admvkl@median.cad/code/public/CT-CLIP

CUDA_VISIBLE_DEVICES=0 python scripts/run_zero_shot.py --checkpoint /mnt/datalake/DS-lake/vankhoa/CT-RATE/models/CT-CLIP-Related/CT-CLIP_v2.pt \
--results-folder /mnt/home/admvkl@median.cad/code/public/CT-CLIP/CT-CLIP_v2 
