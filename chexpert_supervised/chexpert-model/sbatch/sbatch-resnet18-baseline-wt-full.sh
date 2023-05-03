#!/bin/bash
#SBATCH --partition=deep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="r18bwf"
#SBATCH --output=/sailhome/jingbo/CXR_RELATED/exp_logs/r18bwf-%j.out
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
cd ..;

# Semi Ratio = 0.001953125


python train.py \
    --experiment_name resnet18-baseline-wt-full-0.001953125 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 320 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train_semi_0.001953125.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 256 \
    --iters_per_eval 256


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.001953125


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.001953125/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.001953125/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.001953125/final.json

# Semi Ratio = 0.00390625


python train.py \
    --experiment_name resnet18-baseline-wt-full-0.00390625 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 201 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train_semi_0.00390625.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 512 \
    --iters_per_eval 512


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.00390625


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.00390625/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.00390625/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.00390625/final.json

# Semi Ratio = 0.0078125


python train.py \
    --experiment_name resnet18-baseline-wt-full-0.0078125 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 126 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train_semi_0.0078125.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 1024 \
    --iters_per_eval 1024


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.0078125


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.0078125/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.0078125/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.0078125/final.json

# Semi Ratio = 0.015625


python train.py \
    --experiment_name resnet18-baseline-wt-full-0.015625 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 80 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train_semi_0.015625.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 2048 \
    --iters_per_eval 2048


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.015625


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.015625/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.015625/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.015625/final.json

# Semi Ratio = 0.03125


python train.py \
    --experiment_name resnet18-baseline-wt-full-0.03125 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 50 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train_semi_0.03125.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 4096 \
    --iters_per_eval 4096


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.03125


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.03125/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.03125/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.03125/final.json

# Semi Ratio = 0.0625


python train.py \
    --experiment_name resnet18-baseline-wt-full-0.0625 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 31 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train_semi_0.0625.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 8192 \
    --iters_per_eval 8192


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.0625


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.0625/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.0625/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.0625/final.json

# Semi Ratio = 0.125


python train.py \
    --experiment_name resnet18-baseline-wt-full-0.125 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 20 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train_semi_0.125.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 8192 \
    --iters_per_eval 8192


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.125


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.125/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.125/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.125/final.json

# Semi Ratio = 0.25


python train.py \
    --experiment_name resnet18-baseline-wt-full-0.25 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 12 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train_semi_0.25.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 8192 \
    --iters_per_eval 8192


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.25


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.25/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.25/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.25/final.json

# Semi Ratio = 0.5


python train.py \
    --experiment_name resnet18-baseline-wt-full-0.5 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 7 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train_semi_0.5.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 8192 \
    --iters_per_eval 8192


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.5


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.5/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.5/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-0.5/final.json

# Semi Ratio = 1


python train.py \
    --experiment_name resnet18-baseline-wt-full-1 \
    --dataset chexpert_single  \
    --model ResNet18 \
    --num_epochs 5 \
    --metric_name chexpert-competition-AUROC \
    --train_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/train.csv  \
    --val_custom_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/valid.csv \
    --save_dir  ~/CXR_RELATED/chexpert_save \
    --pretrained True \
    --fine_tuning full \
    --ckpt_path None \
    --iters_per_save 8192 \
    --iters_per_eval 8192


python select_ensemble.py \
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \
    --search_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-1


python test.py \
    --dataset custom \
    --moco false \
    --phase test  \
    --together true \
    --ckpt_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-1/best.pth.tar  \
    --save_dir ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-1/test.pth.tar \
    --test_csv /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --test_image_paths /deep/group/data/moco/chexpert-proper-test-4/moving_logs/test.csv \
    --config_path ~/CXR_RELATED/chexpert_save/resnet18-baseline-wt-full-1/final.json


# done
echo "Done"

