#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB              # Partition
#SBATCH --qos=gtx1080ti                # QOS
#SBATCH --job-name=steex_infer_masks_bddoia         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Infer Segmentation Masks BDDOIADB | Started"

python code/infer_masks.py --dataset_name 'BDDOIADB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --data_dir '/nas-ctm01/datasets/public/bdd-oia/data' --metadata_dir '/nas-ctm01/datasets/public/bdd-oia/metadata' --save_dir_masks '/nas-ctm01/datasets/public/bdd-oia/deeplabv3_masks/train' --n_classes 20 --segmentation_network_name 'deeplabv3_bdd10k' --seed 42 --batch_size 1 --subset 'train'
python code/infer_masks.py --dataset_name 'BDDOIADB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --data_dir '/nas-ctm01/datasets/public/bdd-oia/data' --metadata_dir '/nas-ctm01/datasets/public/bdd-oia/metadata' --save_dir_masks '/nas-ctm01/datasets/public/bdd-oia/deeplabv3_masks/val' --n_classes 20 --segmentation_network_name 'deeplabv3_bdd10k' --seed 42 --batch_size 1 --subset 'val'

echo "STEEX-Protocounterfactuals | Infer Segmentation Masks BDDOIADB | Finished"
