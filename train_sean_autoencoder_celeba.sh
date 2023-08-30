#!/bin/bash
#
#SBATCH -p v100_32GB              # Partition
#SBATCH --qos=v100                # QOS
#SBATCH --job-name=steex_train_sean_autoencoder_celeba         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "STEEX-Protocounterfactuals | Train SEAN Autoencoder CelebaDB | Started"

python code/train_sean_autoencoder.py 

echo "STEEX-Protocounterfactuals | Train Decision Model CelebaDB | Finished"
