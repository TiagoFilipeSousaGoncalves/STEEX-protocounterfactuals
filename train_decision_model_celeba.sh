#!/bin/bash
#
#SBATCH -p v100_32GB              # Partition
#SBATCH --qos=v100                # QOS
#SBATCH --job-name=steex_train_decision_model_celeba         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Train Decision Model CelebA | Started"

python code/train_decision_model.py --dataset_name 'CelebaDB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --images_dir '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Img' --images_subdir 'img_align_squared128_celeba' --eval_dir '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Eval' --anno_dir '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Anno' --decision_model_name 'decision_model_celeba' --load_size 128 128 --train_attributes_idx 20 31 39 --batch_size 32 --optimizer 'Adam' --lr 0.0001 --step_size 10 --gamma_scheduler 0.5 --num_epochs 5

echo "STEEX-Protocounterfactuals | Train Decision Model CelebA | Finished"
