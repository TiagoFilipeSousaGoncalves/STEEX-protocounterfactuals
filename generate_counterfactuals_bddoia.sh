#!/bin/bash
#
#SBATCH -p v100_32GB              # Partition
#SBATCH --qos=v100                # QOS
#SBATCH --job-name=steex_generate_counterfactuals_bddoia         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Generate Counterfactuals BDDOIA | Started"

python code/generate_counterfactuals.py --name 'sean_autoencoder_bddoia' --preprocess_mode 'resize_and_crop' --load_size 256 --crop_size 256 --aspect_ratio 1.0 --label_nc 19 --semantic_nc 20 --contain_dontcare_label --dataset_mode 'BDDOIADB' --data_dir '/nas-ctm01/datasets/public/bdd-oia/data' --metadata_dir '/nas-ctm01/datasets/public/bdd-oia/metadata' --masks_dir '/nas-ctm01/datasets/public/bdd-oia/deeplabv3_masks' --split 'val' --gpu_ids '0' --checkpoints_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results/checkpoints' --decision_model_nb_classes 4 --results_dir '/nas-ctm01/datasets/public/bdd-oia/steex_counterfactuals' --how_many 100 --phase 'test' --target_attribute 0 --dataset_name 'BDDOIADB' --decision_model_name 'decision_model_bddoia' --no_instance

echo "STEEX-Protocounterfactuals | Generate Counterfactuals BDDOIA | Finished"
