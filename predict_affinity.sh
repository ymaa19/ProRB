#!/bin/bash

# 定义待遍历的参数
alpha_list=(0.8)
thre_list=(40)

for alpha in "${alpha_list[@]}"; do
  for thre in "${thre_list[@]}"; do
    echo "============================================"
    echo "Processing alpha=${alpha}  thre=${thre}"

    # 预训练权重路径：根据alpha变化
    if [ "$alpha" == "0.8" ]; then
        PRETRAINED="/data/ymxue/p4_protna/code/pretrain_log_review/alpha${alpha}/pretrained_model.pt"
    else
        PRETRAINED="/data/ymxue/p4_protna/code/pretrain_log_new/alpha${alpha}/pretrained_model.pt"
    fi

    # 输出目录：沿用原始 res_${thre}/prorb 结构，并加入 alpha 信息防止覆盖
    OUTPUT_DIR="/data/ymxue/p4_protna/code/B_review/t1_ba/res/res_${thre}/prorb_alpha${alpha}/"
    mkdir -p "${OUTPUT_DIR}"

    python /data/ymxue/p4_protna/code/task1_bind_affinity/finetune_ablation.py \
      --thre ${thre} \
      --alpha ${alpha} \
      --output_folder "${OUTPUT_DIR}" \
      --batch_size 16 \
      --device cuda:7 \
      --num_epochs 100 \
      --lr 1e-4 \
      --pretrained_weights_path PRETRAINED

    echo "Finished alpha=${alpha}  thre=${thre}"
    echo ""
  done
done
