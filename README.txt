# ProRB: A Structure-free Unified Framework for Joint Prediction and Design of Protein-RNA Interactions

This repository contains the official implementation and release of ProRB, a sequence-based dual-modal framework that unifies Protein-RNA joint pretraining, affinity estimation, interface site mapping, and de novo RNA sequence generation.

---

## Repository Structure & Software Release Policy

In this repository, we provide the complete end-to-end training codebase for the foundation pretraining stage (Joint MLM), while providing complete workflows and inference/downstream execution scripts for the downstream predictive and generative tasks.

The directory is structured as follows:

ProRB/
├── model/
│   ├── model.py                 # Core architecture definitions (ProRB, RnaFmModel, etc.)
│   └── attn.py                  # Adaptive Cross-Modal Attention (ProRComAttnModule)
├── utils/
│   ├── dataset.py               # ProteinRNADataset and data processing utilities
│   └── tokenizer.py             # EsmSequenceTokenizer and RnaTokenizer
├── fm_model/
│   ├── rnafm/                   # Local folder for Pretrained RNA-FM weights and config
│   └── esm3/                    # Local folder for Pretrained ESMC weights (ESM-3)
├── requirements.txt             # Python dependencies
├── README.txt                   # Usage documentation (This file)
│
├── 1_pretrain/
│   ├── pretrain.py              # Full joint Protein-RNA Masked Language Modeling training code
│   └── pretrain.sh              # Bash execution script to run pretraining
│
├── 2_binding_affinity/
│   ├── predict_affinity.py      # Core model class and inference execution for affinity tasks
│   └── predict_affinity.sh      # Shell pipeline for evaluating binding affinities
│
├── 3_binding_site/
│   └── predict_bind_site.sh     # Evaluation and downstream execution for binding site mapping
│
└── 4_rna_generation/
    ├── finetune_decoder.py      # Downstream sequence generator fine-tuning script
    ├── finetune_decoder.sh      # Bash workflow to start decoder fine-tuning
    └── generate.py              # Autoregressive RNA generation with temperature scaling

---

## Installation & Environment Setup

We recommend using Anaconda to replicate the exact training and inference environment.

# 1. Create and activate a conda environment
conda create -n prorb python=3.10 -y
conda activate prorb

# 2. Install PyTorch with your local CUDA driver (e.g., CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install other required packages
pip install -r requirements.txt

---

## How to Run and Train the Models

### Step 1: Foundation Pretraining (Full Training Code)
Our core contribution, the dual-modal Joint Masked Language Modeling (MLM), can be trained from scratch using the provided script.

To train the ProRB foundation model:
1. Ensure your training dataset is formatted in a CSV file (containing 'prot_seq' and 'rna_seq' columns).
2. Configure your local data paths and pre-trained language model weight directories (ESM-3/RNA-FM) inside "1_pretrain/pretrain.sh".
3. Run the pretraining pipeline:
   cd 1_pretrain
   bash pretrain.sh

*Technical Note on MLM Loss:*
In our MLM implementation, padding tokens are strictly excluded from the 15% selection mask. During cross-entropy loss computation, all padding and non-masked positions are assigned a label value of -100, which completely bypasses the gradients via PyTorch's ignore_index=-100 mechanism to protect the structural representations from corruption.

---

### Step 2: Downstream Binding Affinity Prediction (Inference Workflow)
For predicting the binding affinity of novel protein-RNA complexes:
1. We provide the complete core model definition and an inference wrapper inside "predict_affinity.py".
2. To run affinity evaluations, specify your CSV data inputs and pre-trained encoder weights in "predict_affinity.sh", then execute:
   cd 2_binding_affinity
   bash predict_affinity.sh

---

### Step 3: Binding Site Interface Mapping (Evaluation Workflow)
To evaluate the cross-modal attention maps and predict interface binding residues for both proteins and RNAs under various sequence identity clusters (e.g., CD-HIT 40% to 90%):
1. Adjust the CD-HIT cluster data directories and output log paths inside "predict_bind_site.sh".
2. Execute the evaluation script:
   cd 3_binding_site
   bash predict_bind_site.sh

---

### Step 4: De Novo RNA Sequence Generation (Decoder Fine-Tuning & Generation)
Our generative pipeline unifies cross-modal fine-tuning and autoregressive sampling.

#### A. Fine-Tuning the Decoder Head
To train the downstream autoregressive generation head and align representations for de novo generation:
1. Specify the path to your pre-trained ProRB model weights in "4_rna_generation/finetune_decoder.sh".
2. Run the fine-tuning script:
   cd 4_rna_generation
   bash finetune_decoder.sh

#### B. Autoregressive Sequence Generation
After fine-tuning, run the interactive generation module to design candidate RNA sequences for target proteins across multiple sampling temperatures (e.g., 0.4 to 1.6):
python generate.py

---

## Paths Configuration Note
To ensure 100% scientific reproducibility of the benchmarks reported in our manuscript, some paths in the scripts are pre-configured to local clusters. Users and developers should modify the following variables in the .sh and .py files to match their own local directories:

- --pretrained_weights_path: Directory of the foundation model checkpoints.
- csv_file / data_folder: Directories containing train/val/test CSV splits.
- --model_save_path / --output_folder: Desired directories for saving training logs and checkpoints.
