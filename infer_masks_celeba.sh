#!/bin/bash
#
#SBATCH -p a100_80GB              # Partition
#SBATCH --qos=a100                # QOS
#SBATCH --job-name=steex_infer_masks_celeba         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Infer Segmentation Masks CelebA | Started"

python code/infer_masks.py --dataset_name 'CelebaDB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --images_dir '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Img' --images_subdir 'img_align_squared128_celeba' --eval_dir '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Eval' --anno_dir '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Anno' --save_dir_masks '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Masks-DeepLabV3' --n_classes 19 --segmentation_network_name 'deeplabv3_celebamaskhq' --seed 42 --batch_size 1 --subset 'train'
python code/infer_masks.py --dataset_name 'CelebaDB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --images_dir '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Img' --images_subdir 'img_align_squared128_celeba' --eval_dir '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Eval' --anno_dir '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Anno' --save_dir_masks '/nas-ctm01/datasets/public/BIOMETRICS/celeba-db/Masks-DeepLabV3' --n_classes 19 --segmentation_network_name 'deeplabv3_celebamaskhq' --seed 42 --batch_size 1 --subset 'val'

echo "STEEX-Protocounterfactuals | Infer Segmentation Masks CelebA | Finished"
