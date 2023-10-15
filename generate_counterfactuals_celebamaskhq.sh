#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB              # Partition
#SBATCH --qos=gtx1080ti                # QOS
#SBATCH --job-name=steex_generate_counterfactuals_celebamaskhq        # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Generate Counterfactuals CelebaMaskHQDB | Started"

python code/generate_counterfactuals.py --name 'sean_autoencoder_celebamaskhq' --preprocess_mode 'scale_width_and_crop' --load_size 256 --crop_size 256 --aspect_ratio 1.0 --label_nc 18 --semantic_nc 19 --contain_dontcare_label --dataset_mode 'CelebaMaskHQDB' --images_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/CelebA-HQ-img' --eval_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/Eval' --anno_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/Anno' --masks_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/CelebAMaskHQ-mask-deeplabv3' --split 'test' --gpu_ids '0' --checkpoints_dir '/nas-ctm01/homes/tgoncalv/STEEX-protocounterfactuals/results/checkpoints' --decision_model_nb_classes 3 --results_dir '/nas-ctm01/datasets/public/BIOMETRICS/celebamask-hq-db/steex_counterfactuals' --phase 'test' --target_attribute 1 --dataset_name 'CelebaMaskHQDB' --decision_model_name 'decision_model_celebamaskhq' --no_instance --save_query_image --save_reconstruction --save_initial_final_z --remove_saved_style_codes --nThreads 0

echo "STEEX-Protocounterfactuals | Generate Counterfactuals CelebaMaskHQDB | Finished"
