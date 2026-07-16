python /data/ymxue/p4_protna/code/pretrain.py \
    --batch_size 16 \
    --lr 1e-4 \
    --num_epochs 40 \
    --device cuda:7 \
    --folder /data/ymxue/p4_protna/code/pretrain_log_review/alpha0.8 \
    --log_file /data/ymxue/p4_protna/code/pretrain_log_review/alpha0.8/pretrain.txt \
    --alpha 0.8 \
    --model_save_path /data/ymxue/p4_protna/code/pretrain_log_review/alpha0.8/pretrained_model.pt 
