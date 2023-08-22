#!/bin/bash
#
#SBATCH -p a100_80GB              # Partition
#SBATCH --qos=a100                # QOS
#SBATCH --job-name=steex_train_segmentation_model_bdd100k         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Train Segmentation Model BDD100kDB | Started"

python code/train_segmentation_model.py --dataset_name 'BDD100kDB' --results_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results' --images_dir '/nas-ctm01/datasets/public/bdd100k/images' --labels_dir '/nas-ctm01/datasets/public/bdd100k/labels' --n_classes 20 --segmentation_network_name 'deeplabv3_bdd100k' --seed 42 --batch_size 8 --num_epochs 50

echo "STEEX-Protocounterfactuals | Train Segmentation Model BDD100kDB | Finished"
