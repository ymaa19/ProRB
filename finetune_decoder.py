import sys
sys.path.append('/data/ymxue/p4_protna/code/')

from utils.dataset import *
from utils.tokenizer import *
from model.model import *
from model.attn import *

# 下游任务头
class DownstreamHead(nn.Module):
    def __init__(self, input_dim):
        super(DownstreamHead, self).__init__()
        self.non_lin = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 26)
        )
    def forward(self, x):
        token_scores = self.non_lin(x) 
        return token_scores

from pretrain import Model
# 微调模型
class ModelFineTune(nn.Module):
    def __init__(self, alpha, hidden_dim=64, pretrained_weights_path=None):
        # 初始化模型
        super(ModelFineTune, self).__init__()
        # 定义模型
        self.model = Model(hidden_dim, alpha=alpha)
        # 定义下游任务头
        self.head = DownstreamHead(hidden_dim)
        # self.token_emb = nn.Embedding(26, hidden_dim)

        # 加载预训练权重
        if pretrained_weights_path is not None:
            self.load_pretrained_weights(pretrained_weights_path)
        
        valid_tokens = [1, 2, 4, 6, 7, 8, 9]
        mask = torch.full((26,), -1e4) # 用一个极小的数填充
        mask[valid_tokens] = 0
        self.register_buffer('logit_mask', mask) # 注册为 buffer，随模型移动到 GPU

    def load_pretrained_weights(self, path):
        """
        加载预训练权重。
        :param path: 预训练权重文件路径
        """
        try:
            pretrained_weights = torch.load(path, map_location='cpu')
            # keys_log_path = "/data/ymxue/p4_protna/code/A_review/t3_gen/debug_1_model_keys_load_miss.txt"
            
            processed_weights = {}
            for k, v in pretrained_weights.items():
                if k.startswith("model."):
                    new_key = k[6:] # 跳过开头的 "model."
                else:
                    new_key = k
                processed_weights[new_key] = v
        
            # 执行加载
            self.model.load_state_dict(processed_weights, strict=False)
            
            # with open(keys_log_path, "w") as f:
            #     # 新增：打印模型原本应该有的所有 Key
            #     f.write("=== [0] Current Model Keys (模型原本期望的所有 Key) ===\n")
            #     for k in self.model.state_dict().keys():
            #         f.write(f"{k}\n")
                
            #     f.write("\n" + "="*50 + "\n")
            #     f.write("=== [1] Processed Pretrained Keys (你处理后准备喂给模型的 Key) ===\n")
            #     for k in processed_weights.keys():
            #         f.write(f"{k}\n")
                
            #     f.write("\n" + "="*50 + "\n")
            #     f.write("=== [2] Missing Keys (模型想要，但你没给成的) ===\n")
            #     if msg.missing_keys:
            #         for k in msg.missing_keys:
            #             f.write(f"{k}\n")
            #     else:
            #         f.write("None\n")
                    
            #     f.write("\n" + "="*50 + "\n")
            #     f.write("=== [3] Unexpected Keys (你给了，但模型不想要的) ===\n")
            #     for k in msg.unexpected_keys:
            #         f.write(f"{k}\n")

            # print(f"Debug log saved to: {keys_log_path}")
            
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

            
    def forward(self, protein_input, rna_input, prot_pad=None, rna_pad=None):
        # print("forward")
        _, _, _, auxloss, fake_rna = self.model(rna_input, protein_input, prot_pad, rna_pad, mode="generate")
        token_level = self.head(fake_rna)
        token_level = token_level + self.logit_mask
        # print("token_level shape:", token_level.shape)  # 调试输出
        return token_level, auxloss, fake_rna

# 训练函数
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, average_precision_score, f1_score

def train(model, dataloader, optimizer, criterion, device, log_file):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        prot_input_ids = batch["prot_input_ids"].to(device)
        rna_input_ids = batch["rna_input_ids"].to(device)
        prot_padding_mask = batch["prot_padding_mask"].to(device)
        rna_labels = batch["rna_labels"].to(device)
        rna_padding_mask = batch["rna_padding_mask"].to(device)

        optimizer.zero_grad()
        token_level, auxloss, fake_rna = model(prot_input_ids, rna_input_ids, prot_pad=prot_padding_mask, rna_pad=rna_padding_mask)
        pred = token_level.contiguous().view(-1, 26)
        tgt = rna_labels.contiguous().view(-1)
        cross_loss = criterion(pred, tgt)
        loss = cross_loss
        
        # 打印损失
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item()} CrossLoss = {cross_loss.item()}")
            with open(log_file, "a") as f:
                f.write(f"Batch {batch_idx}: Loss = {loss.item()} CrossLoss = {cross_loss.item()}\n")
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # print(f"token_level: {token_level}")
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, output_path="/data/ymxue/p4_protna/code/A_review/t3_gen/debug1_val_results.txt"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # 准备写入结果的文件
    with open(output_path, "a") as f_out:
        # f_out.write(f"\n--- Epoch {epoch + 1} Validation Results ---\n")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                prot_input_ids = batch["prot_input_ids"].to(device)
                rna_input_ids = batch["rna_input_ids"].to(device)
                prot_padding_mask = batch["prot_padding_mask"].to(device)
                rna_labels = batch["rna_labels"].to(device)
                rna_padding_mask = batch["rna_padding_mask"].to(device)

                token_level, auxloss, fake_rna = model(prot_input_ids, rna_input_ids, 
                                             prot_pad=prot_padding_mask, rna_pad=rna_padding_mask)
                
                pred = token_level.contiguous().view(-1, 26)
                tgt = rna_labels.contiguous().view(-1)
                
                # 过滤掉 ignore_index=-100
                mask = (tgt != -100)
                if not mask.any():
                    continue
                
                c_loss = criterion(pred, tgt)
                loss = c_loss 
                total_loss += loss.item()

                # 获取预测值
                pred_ids = pred.argmax(dim=-1)
                
                # 提取当前 batch 中有效的预测和标签进行记录
                # 为了可读性，按 sequence 恢复并记录
                batch_preds = pred_ids[mask].cpu().numpy()
                batch_targets = tgt[mask].cpu().numpy()
                
                # f_out.write(f"Batch {batch_idx}:\n")
                # f_out.write(f"  Target: {batch_targets.tolist()}\n")
                # f_out.write(f"  Predict: {fake_rna.tolist()}\n")

                all_preds.append(batch_preds)
                all_targets.append(batch_targets)

    avg_loss = total_loss / len(dataloader)
    if all_preds:
        acc = accuracy_score(np.concatenate(all_targets), np.concatenate(all_preds))
    else:
        acc = 0.0
        
    return avg_loss, acc

from tqdm import tqdm
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
# 主程序
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Train and validate a fine-tuned model.")

    # 数据相关参数
    parser.add_argument("--max_prot_len", type=int, default=1024, help="Maximum length of protein sequences.")
    parser.add_argument("--max_rna_len", type=int, default=512, help="Maximum length of RNA sequences.")

    # 模型相关参数
    parser.add_argument("--vocab_size", type=int, default=64, help="Vocabulary size.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of the model.")
    parser.add_argument("--pretrained_weights_path", type=str, default=None, help="Path to pretrained weights.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha value for the model.")

    # 训练相关参数
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--log_file", type=str, default="finetune_log.txt", help="Log file name.")
    parser.add_argument("--model_save_path", type=str, default="finetuned_model.pth", help="Path to save the trained model.")

    args = parser.parse_args()

    # tokenizer = CustomTokenizer()
    # tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})
    # 数据集路径
    csv_file = "/data/ymxue/p4_protna/code/A_review/t3_gen/d1_1_19_cdhit90/train.csv"
    dataset = finetune_decoder(csv_file, 
                                max_prot_len=1024,
                                max_rna_len=512)  
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_csv = "/data/ymxue/p4_protna/code/A_review/t3_gen/d1_1_19_cdhit/val.csv" # 确保路径正确
    val_dataset = finetune_decoder(val_csv, max_prot_len=1024, max_rna_len=512)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ModelFineTune(alpha=args.alpha, hidden_dim=64, pretrained_weights_path=args.pretrained_weights_path).to(args.device)
    # freeze parameters of the model.model
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.rna_norm_gen.parameters():
        param.requires_grad = True
    for param in model.model.token_emb.parameters():
        param.requires_grad = True
    for param in model.model.encoder_layer.parameters():
        param.requires_grad = True
    for param in model.model.prorcom_attn.rna_decoder.parameters():
        param.requires_grad = True
    for param in model.model.prot_norm.parameters():
        param.requires_grad = True
    model.model.pos_emb.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    valid_tokens = [1, 2, 4, 6, 7, 8, 9]

    # 初始化权重为 0
    weights = torch.zeros(26).to(args.device)

    # 给有效 token 赋值
    weights[valid_tokens] = 1.0 
    weights[2] = 5.0 # 进一步加大终止符权重（如果这是你的需求）

    # 这里的 ignore_index 依然保留 -100 处理 padding
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100, label_smoothing=0.1)

    best_val_loss = float('inf')  # 初始化为无穷大
    for epoch in range(args.num_epochs):
        # 1. 训练
        train_avg_loss = train(model, dataloader, optimizer, criterion, device=args.device, log_file=args.log_file)
        
        # 2. 验证
        val_avg_loss, val_acc = validate(model, val_dataloader, criterion, device=args.device)
        
        # 3. 打印和记录
        log_str = (f"Epoch {epoch + 1}/{args.num_epochs} | "
                   f"Train Loss: {train_avg_loss:.4f} | "
                   f"Val Loss: {val_avg_loss:.4f} | "
                   f"Val Acc: {val_acc:.4f}")
        print(log_str)
        with open(args.log_file, "a") as f:
            f.write(log_str + "\n")

        # 4. 根据验证集 Loss 保存权重
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"--- Best model saved at Epoch {epoch + 1} (Val Loss: {val_avg_loss:.4f}) ---")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} LR: {current_lr:.2e}")
