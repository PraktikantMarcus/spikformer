#!/bin/bash
# SBATCH --job-name=simple-imagenet-download
# SBATCH --partition=normal
# SBATCH --exclude=multigpu
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --time=0-05:00:00
# SBATCH --output=logs/simple_down_%j.out
# SBATCH --error=logs/simple_down_%j.err


%--------------------Environment------------------------------

source /home/fritzsche/qkformer/bin/activate

mkdir -p logs

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

wget -c --show-progress -P /nfsscratch/fritzsche/imagenet \
    https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar \
    https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar