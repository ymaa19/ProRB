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
        self.token_emb = nn.Embedding(26, hidden_dim)

        # 加载预训练权重
        if pretrained_weights_path is not None:
            self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, path):
        """
        加载预训练权重。
        :param path: 预训练权重文件路径
        """
        try:
            pretrained_weights = torch.load(path, map_location='cpu')
            if "model" in pretrained_weights:
                pretrained_weights = pretrained_weights["model"]
            pretrained_weights = {k.replace("model.", ""): v for k, v in pretrained_weights.items()}
            self.model.load_state_dict(pretrained_weights, strict=False)
            print("Successfully loaded pretrained weights.")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    def forward(self, protein_input, rna_input, prot_pad=None, rna_pad=None):
        # print("forward")
        _, _, _, auxloss, fake_rna = self.model(rna_input, protein_input, prot_pad, rna_pad, mode="generate")
        token_level = self.head(fake_rna[0])
        return token_level, auxloss

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
        token_level, auxloss = model(prot_input_ids, rna_input_ids, prot_pad=prot_padding_mask, rna_pad=rna_padding_mask)
        pred = token_level.contiguous().view(-1, 26)
        tgt = rna_labels.contiguous().view(-1)
        loss = criterion(pred, tgt)
        
        # 打印损失
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item()}")
            with open(log_file, "a") as f:
                f.write(f"Batch {batch_idx}: Loss = {loss.item()}\n")
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

from tqdm import tqdm
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
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
    csv_file = "/data/ymxue/p4_protna/code/task2_bindingsite/data_trunc_rna_mfe/cdhit70/train_filtered.csv"
    dataset = finetune_decoder(csv_file, 
                                max_prot_len=1024,
                                max_rna_len=512)  
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = ModelFineTune(alpha=args.alpha, hidden_dim=64, pretrained_weights_path=args.pretrained_weights_path).to(args.device)
    # freeze parameters of the model.model
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.prot_norm.parameters():
        param.requires_grad = True
    for param in model.model.rna_norm.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充位置

    loss_min = 100000
    # 训练模型
    for epoch in range(args.num_epochs):
        avg_loss = train(model, dataloader, optimizer, criterion, device=args.device, log_file=args.log_file)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")
        with open(args.log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}\n")
        if avg_loss < loss_min:
            loss_min = avg_loss
            # 保存模型
            torch.save(model.state_dict(), args.model_save_path)

