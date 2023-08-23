#!/bin/bash
#
#SBATCH -p a100_80GB              # Partition
#SBATCH --qos=a100                # QOS
#SBATCH --job-name=steex_infer_masks_celebamaskhq         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Infer Masks CelebaMaskHQDB | Started"

python code/infer_masks.py --dataset_name 'CelebaMaskHQDB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --images_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/CelebA-HQ-img' --masks_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/CelebAMaskHQ-mask' --eval_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/Eval' --anno_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/Anno' --n_classes 19 --segmentation_network_name 'deeplabv3_celebamaskhq' --seed 42 --batch_size 8

echo "STEEX-Protocounterfactuals | Infer Masks CelebaMaskHQDB | Finished"
