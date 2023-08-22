#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB              # Partition
#SBATCH --qos=gtx1080ti                # QOS
#SBATCH --job-name=steex_train_decision_model_bddoia         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Train Decision Model CelebaMaskHQDB | Started"

python train_decision_model.py --dataset_name 'BDDOIADB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --data_dir '/nas-ctm01/datasets/public/bdd-oia/data' --metadata_dir '/nas-ctm01/datasets/public/bdd-oia/metadata' --decision_model_name 'decision_model_bddoia' --crop_size 512 256 --train_attributes_idx 0 1 2 3 --batch_size 8 --optimizer 'Adam' --lr 0.0001 --step_size 10 --gamma_scheduler 0.5 --num_epochs 5

echo "STEEX-Protocounterfactuals | Train Decision Model CelebaMaskHQDB | Finished"
