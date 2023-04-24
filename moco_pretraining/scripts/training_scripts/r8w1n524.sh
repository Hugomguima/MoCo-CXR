#!/bin/bash
#SBATCH --partition=deep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="r8w1n524"
#SBATCH --output=exp_logs/r8w1n524-%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample job
NPROCS=`sbatch --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

cd ../moco; python main_moco.py -a resnet18 \
            --lr 1e-05 --batch-size 24 \
            --epochs 20S \
            --world-size 1 --rank 0 \
            --mlp --moco-t 0.2 --from-imagenet \
            --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
			--aug-setting chexpert --rotate 10 --maintain-ratio \
            --train_data /deep/group/data/moco/chexpert-proper-test/data/full_train \
            --exp-name r8w1n524_20230420h15

# done
echo "Done"
