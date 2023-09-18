#!/bin/bash
#
#SBATCH -p a100_80GB              # Partition
#SBATCH --qos=a100                # QOS
#SBATCH --job-name=steex_train_sean_autoencoder_bddoia         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Train SEAN Autoencoder BDDOIADB | Started"

python code/train_sean_autoencoder.py --name 'sean_autoencoder_bddoia' --gpu_ids '0' --checkpoints_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results/checkpoints' --phase 'train' --batchSize 16 --preprocess_mode 'resize_and_crop' --load_size 256 --crop_size 256 --aspect_ratio 1.0 --label_nc 19 --semantic_nc 20 --contain_dontcare_label --dataset_mode 'BDDOIADB' --data_dir '/nas-ctm01/datasets/public/bdd-oia/data' --metadata_dir '/nas-ctm01/datasets/public/bdd-oia/metadata' --masks_dir '/nas-ctm01/datasets/public/bdd-oia/deeplabv3_masks' --nThreads 4 --augment --no_instance --no_html

echo "STEEX-Protocounterfactuals | Train Decision Model BDDOIADB | Finished"
