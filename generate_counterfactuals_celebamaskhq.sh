#!/bin/bash
#
#SBATCH -p gtx1080_8GB              # Partition
#SBATCH --qos=gtx1080                # QOS
#SBATCH --job-name=steex_preproc_celeba         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Preprocess CelebA | Started"

python code/data_resize.py --in_folder '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Img/img_align_celeba' --out_folder '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Img/img_align_squared128_celeba' --new_size 128

echo "STEEX-Protocounterfactuals | Preprocess CelebA | Finished"
