import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 假设这些是你自己的工具类
from utils.dataset import DownstreamDS_token
from model.model import * 
from pretrain import Model # 引用基础模型

# --- 精简后的推理模型类 ---

class DownstreamHead(nn.Module):
    def __init__(self, input_dim):
        super(DownstreamHead, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 推理时输出原始 logits
        return self.fc(x).squeeze(-1)

class ModelInference(nn.Module):
    def __init__(self, alpha=0.1, hidden_dim=64, weights_path=None, device="cuda"):
        super(ModelInference, self).__init__()
        self.model = Model(hidden_dim, alpha=alpha)
        self.rna_head = DownstreamHead(hidden_dim)
        self.prot_head = DownstreamHead(hidden_dim)
        self.device = device

        if weights_path:
            self.load_weights(weights_path)
        self.to(device)
        self.eval()

    def load_weights(self, path):
        """ 加载微调后的模型权重 """
        try:
            state_dict = torch.load(path, map_location='cpu')
            # 自动处理 DDP 保存时产生的 'module.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
            print(f"Successfully loaded weights from {path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    @torch.no_grad()
    def forward(self, protein_input, rna_input, prot_pad, rna_pad):
        # mode='val' 通常用于推理，避开 pretrain 模型中的 Dropout 或随机采样
        _, rna_embed, protein_embed, _, _ = self.model(
            rna_input, protein_input, prot_pad, rna_pad, mode='val'
        )
        
        # 获取 Logits 并转换为概率
        rna_logits = self.rna_head(rna_embed)
        prot_logits = self.prot_head(protein_embed)
        
        rna_probs = torch.sigmoid(rna_logits)
        prot_probs = torch.sigmoid(prot_logits)
        
        return rna_probs, prot_probs

# --- 推理执行脚本 ---

def run_prediction(data_csv, weights_path, output_path, device="cuda"):
    # 1. 初始化推理模型
    inf_model = ModelInference(alpha=0.1, weights_path=weights_path, device=device)

    # 2. 准备数据集 (推理不需要 Sampler)
    dataset = DownstreamDS_token(data_csv)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    results = []

    print("Starting Inference...")
    for batch in dataloader:
        prot_ids = batch["prot_input_ids"].to(device)
        rna_ids = batch["rna_input_ids"].to(device)
        prot_mask = batch["prot_padding_mask"].to(device)
        rna_mask = batch["rna_padding_mask"].to(device)

        # 执行预测
        rna_site_probs, prot_site_probs = inf_model(prot_ids, rna_ids, prot_mask, rna_mask)

        # 保存结果 (转为 list 或 numpy)
        results.append({
            "rna_probs": rna_site_probs.cpu().numpy(),
            "prot_probs": prot_site_probs.cpu().numpy()
        })

    # 3. 保存推理结果到磁盘
    torch.save(results, output_path)
    print(f"Inference complete. Results saved to {output_path}")

if __name__ == '__main__':
    # 填入你的路径
    DATA_PATH = "/path/to/your/test.csv"
    CHECKPOINT = "/path/to/your/best_model.pt"
    SAVE_FILE = "./prediction_results.pth"
    
    run_prediction(DATA_PATH, CHECKPOINT, SAVE_FILE)
