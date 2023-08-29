#!/bin/bash
#
#SBATCH -p v100_32GB              # Partition
#SBATCH --qos=v100                # QOS
#SBATCH --job-name=steex_infer_masks_bdd10k         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Infer Segmentation Masks BDD10kDB | Started"

python code/infer_masks.py --dataset_name 'BDD10kDB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --images_dir '/nas-ctm01/datasets/public/bdd100k/images' --labels_dir '/nas-ctm01/datasets/public/bdd100k/labels' --save_dir_masks '/nas-ctm01/datasets/public/bdd100k/labels/sem_seg_deeplabv3/val/' --n_classes 20 --segmentation_network_name 'deeplabv3_bdd10k' --seed 42 --batch_size 16

echo "STEEX-Protocounterfactuals | Infer Segmentation Masks BDD10kDB | Finished"