python /data/ymxue/p4_protna/code/A_review/t3_gen/p1_train_gen.py \
    --batch_size 8 \
    --lr 1e-4 \
    --num_epochs 80 \
    --device cuda:0 \
    --log_file /data/ymxue/p4_protna/code/A_review/t3_gen/log1_train_gen_refine.txt \
    --alpha 0.8 \
    --model_save_path /data/ymxue/p4_protna/code/A_review/t3_gen/pt1_train_gen_refine.pt \
    --pretrained_weights_path /data/ymxue/p4_protna/code/pretrain_log_review/alpha0.8/pretrained_model.pt
