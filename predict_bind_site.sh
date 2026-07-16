#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate p3text

# 设置可见显卡
export CUDA_VISIBLE_DEVICES=1

# --- 定义需要遍历的参数 ---
# seeds=(19 20 21)               # 在这里添加你想跑的 seed
seeds=(8 17 19 21 42)
alphas=(0.8)           # 在这里添加你想跑的 alpha
cdhit_vals=(40 50 60 70 80 90) # 对应的 cdhit 阈值  

# 外层循环：alpha
for alpha in "${alphas[@]}"; do
    # 中层循环：seed
    for seed in "${seeds[@]}"; do
        # 内层循环：cdhit 阈值
        for val in "${cdhit_vals[@]}"; do

            # 动态生成输入和输出路径
            data_folder="/data/ymxue/p4_protna/code/A_review/t2_bs/d1_3_cdhit/seed_${seed}/cdhit${val}/"
            output_folder="/data/ymxue/p4_protna/code/A_review/t2_bs/log2_train_test/seed_${seed}/cdhit${val}/prorattn_no_cross_attn_${alpha}/"
            
            # 根据 alpha 的值选择不同的预训练路径
            if [ "$alpha" == "0.8" ]; then
                pretrained_weights="/data/ymxue/p4_protna/code/pretrain_log_review/alpha${alpha}/pretrained_model.pt"
            else
                pretrained_weights="/data/ymxue/p4_protna/code/pretrain_log_new/alpha${alpha}/pretrained_model.pt"
            fi

            echo "------------------------------------------------"
            echo "Running: Seed=$seed, Alpha=$alpha, CD-HIT=$val"
            echo "Data: $data_folder"
            echo "Output: $output_folder"
            
            # 如果输出目录不存在，则创建（可选，增强脚本鲁棒性）
            mkdir -p "$output_folder"

            # 调用 Python 脚本
            torchrun --nproc_per_node=1 --master_port=29501 /data/ymxue/p4_protna/code/A_review/t2_bs/p2_train_test.py \
                --data_folder "$data_folder" \
                --output_folder "$output_folder" \
                --batch_size 16 \
                --num_epochs 60 \
                --lr 1e-4 \
                --alpha "$alpha" \
                # --pretrained_weights_path "None" \
                --pretrained_weights_path "$pretrained_weights"
        done
    done
done
