#! /bin/bash
#SBATCH --nodes=1
#SBATCH -J mednist_train
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH -p big

set -v

# change this if run submitted from a different directory
export BUNDLE="$(pwd)/.."

# have to set PYTHONPATH to find MONAI and GenerativeModels as well as the bundle's script directory
export PYTHONPATH="$HOME/MONAI:$HOME/GenerativeModels:$BUNDLE"

# change this to load a checkpoint instead of started from scratch
CKPT=none

CONFIG="'$BUNDLE/configs/common.yaml', '$BUNDLE/configs/train.yaml', '$BUNDLE/configs/train_multigpu.yaml'"

# change this to point to where MedNIST is located
DATASET="$(pwd)"

# it's useful to include the configuration in the log file
cat "$BUNDLE/configs/common.yaml"
cat "$BUNDLE/configs/train.yaml"
cat "$BUNDLE/configs/train_multigpu.yaml"

# remember to change arguments to match how many nodes and GPUs you have
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --config_file "$CONFIG" \
    --logging_file "$BUNDLE/configs/logging.conf" \
    --bundle_root "$BUNDLE" \
    --dataset_dir "$DATASET"
