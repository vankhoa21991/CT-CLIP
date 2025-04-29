#!/bin/bash
##File automaticaly generated on Mon Jun 14 16:26:53 2021
#SBATCH --job-name=trCLIP
#SBATCH --time=1000:00:00
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=26
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
conda activate venv_py311

cd /mnt/home/admvkl@median.cad/code/public/CT-CLIP
# python download_only_train_data.py 
# python download_only_valid_data.py

# cd /mnt/home/admvkl@median.cad/code/public/CT-CLIP
python /mnt/home/admvkl@median.cad/code/public/CT-CLIP/data_preprocess/preprocess_ctrate_train.py

python /mnt/home/admvkl@median.cad/code/public/CT-CLIP/data_preprocess/preprocess_ctrate_valid.py