#!/bin/bash

#SBATCH --job-name=dinosaur
#SBATCH --output=logs/dinosaur.%j.out
#SBATCH --error=logs/dinosaur.%j.err
#SBATCH -A uli@v100
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH -t 20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL

echo ${SLURM_NODELIST}

source ~/.bashrc

module purge
module load pytorch-gpu/py3/2.0.0

export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=.
export WANDB_MODE=offline

data_dir="./data/COCO/"
output_dir="./output/dinosaur"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 13000 --nproc_per_node=1 \
    train_dinoslot.py \
    --which_encoder dino_vitb16 \
    --dataset coco \
    --data_path ${data_dir} \
    --epochs 30 \
    --num_slots 7 \
    --train_permutations standard \
    --log_path ${output_dir}
