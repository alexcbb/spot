#!/bin/bash
#SBATCH --job-name=dinosaur
#SBATCH --output=logs/dinosaur/dinosaur.%j.out
#SBATCH --error=logs/dinosaur/dinosaur.%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 80:00:00
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL

echo ${SLURM_NODELIST}

source ~/.bashrc

conda activate mttoc

data_dir="./data/COCO/"
output_dir="./output/dinosaur"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 13000 --nproc_per_node=1 \
    train_spot.py \
    --dataset coco \
    --data_path ${data_dir} \
    --epochs 100 \
    --num_slots 7 \
    --train_permutations standard \
    --log_path ${output_dir}
