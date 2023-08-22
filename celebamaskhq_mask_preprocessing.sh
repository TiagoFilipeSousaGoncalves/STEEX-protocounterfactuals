#!/bin/bash
#
#SBATCH -p debug              # Partition
#SBATCH --qos=debug                # QOS
#SBATCH --job-name=steex_preproc_celebamaskhq         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Preprocess CelebAMask | Started"

python code/celebamaskhq_mask_preprocessing.py --folder_base '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/CelebAMask-HQ-mask-anno' --folder_save '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/CelebAMaskHQ-mask'

echo "STEEX-Protocounterfactuals | Preprocess CelebAMask | Finished"
