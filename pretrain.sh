#!/usr/bin/env bash
set -euo pipefail

# Configure these paths for your installation. DATA_FOLDER must contain
# train.csv and test.csv with columns: prot,rna,prot_bind,rna_bind.
DATA_FOLDER=${DATA_FOLDER:-/path/to/cdhit40}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-./outputs/binding_site/cdhit40_seed42}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-/path/to/pretrained_model.pt}
RNAFM_PATH=${RNAFM_PATH:-/path/to/rnafm}
ESMC_WEIGHTS_PATH=${ESMC_WEIGHTS_PATH:-/path/to/esmc_300m_2024_12_v0.pth}
DEVICE=${DEVICE:-cuda:0}

python p2_train_test.py \
  --data_folder "$DATA_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --pretrained_weights_path "$PRETRAINED_WEIGHTS" \
  --rnafm_path "$RNAFM_PATH" \
  --esmc_weights_path "$ESMC_WEIGHTS_PATH" \
  --device "$DEVICE" \
  --batch_size 16 \
  --num_epochs 60 \
  --lr 1e-4 \
  --alpha 0.8 \
  --pos_weight 15 \
  --seed 42
