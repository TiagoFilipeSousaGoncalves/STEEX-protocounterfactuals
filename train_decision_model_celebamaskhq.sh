#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB              # Partition
#SBATCH --qos=gtx1080ti                # QOS
#SBATCH --job-name=steex_train_decision_model_celebamaskhq         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Train Decision Model CelebaMaskHQDB | Started"

python train_decision_model.py --dataset_name 'CelebaMaskHQDB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --images_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/CelebA-HQ-img' --eval_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/Eval' --anno_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/Anno' --decision_model_name 'decision_model_celebamaskhq' --load_size 256 256 --train_attributes_idx 20 31 39 --batch_size 32 --optimizer 'Adam' --lr 0.0001 --step_size 10 --gamma_scheduler 0.5 --num_epochs 5

echo "STEEX-Protocounterfactuals | Train Decision Model CelebaMaskHQDB | Finished"
