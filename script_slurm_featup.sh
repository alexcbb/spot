#!/bin/bash

#SBATCH --job-name=dinosaur_featup
#SBATCH --output=logs/dinosaur_featup.%j.out
#SBATCH --error=logs/dinosaur_featup.%j.err
#SBATCH -A uli@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 30:00:00
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL

echo ${SLURM_NODELIST}

source ~/.bashrc

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0

export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=.

data_dir="./data/COCO/"
output_dir="./output/dinosaur"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 13000 --nproc_per_node=1 \
    train_dinoslot.py \
    --which_encoder featup_dino16 \
    --dataset coco \
    --data_path ${data_dir} \
    --batch_size 128 \
    --epochs 30 \
    --num_slots 7 \
    --train_permutations standard \
    --log_path ${output_dir}
