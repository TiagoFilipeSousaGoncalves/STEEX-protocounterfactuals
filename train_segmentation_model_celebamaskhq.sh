#!/bin/bash
#
#SBATCH -p v100_32GB              # Partition
#SBATCH --qos=v100                # QOS
#SBATCH --job-name=steex_train_segmentation_model_celebamaskhq         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Train Segmentation Model CelebaMaskHQDB | Started"

python code/train_segmentation_model.py --dataset_name 'CelebaMaskHQDB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --images_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/CelebA-HQ-img' --masks_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/CelebAMaskHQ-mask' --eval_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/Eval' --anno_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/Anno' --n_classes 19 --segmentation_network_name 'deeplabv3_celebamaskhq' --seed 42 --batch_size 8 --num_epochs 50

echo "STEEX-Protocounterfactuals | Train Segmentation Model CelebaMaskHQDB | Finished"
