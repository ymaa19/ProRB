import sys
sys.path.append('/data/ymxue/p4_protna/code/')

from utils.dataset import *
from utils.tokenizer import *
from model.model import *
from model.attn import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 下游任务头
class DownstreamHead(nn.Module):
    def __init__(self, input_dim):
        super(DownstreamHead, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        token_scores = self.fc(x).squeeze(-1)  
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
        self.rna_head = DownstreamHead(hidden_dim)
        self.prot_head = DownstreamHead(hidden_dim)

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
            processed_weights = {}
            for k, v in pretrained_weights.items():
                if k.startswith("model."):
                    new_key = k[6:] # 跳过开头的 "model."
                else:
                    new_key = k
                processed_weights[new_key] = v
            self.model.load_state_dict(processed_weights, strict=False)
            print("Successfully loaded pretrained weights.")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    def forward(self, protein_input, rna_input=None, prot_pad=None, rna_pad=None):
        # print("forward")
        complex_embed, rna_embed, protein_embed, aux_loss, fake_rna = self.model(rna_input, protein_input, prot_pad, rna_pad, mode='train')
        # print("forward1")
        res_rna = self.rna_head(rna_embed)
        res_protein = self.prot_head(protein_embed)
        return res_rna, res_protein, (rna_embed, protein_embed, fake_rna)

# 训练函数
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, average_precision_score, f1_score
import random

def train(model, dataloader, optimizer, device, local_rank):
    model.train()
    total_loss = 0
    res_lis_rna = []
    res_lis_prot = []
    tgt_lis_rna = []
    tgt_lis_prot = []

    for batch_idx, batch in enumerate(dataloader):
        prot_input_ids = batch["prot_input_ids"].to(device)
        rna_input_ids = batch["rna_input_ids"].to(device)
        prot_bind = batch["prot_bind"].to(device)
        rna_bind = batch["rna_bind"].to(device)
        prot_padding_mask = batch["prot_padding_mask"].to(device)
        rna_padding_mask = batch["rna_padding_mask"].to(device)

        optimizer.zero_grad()
        res_rna, res_prot, _ = model(rna_input=rna_input_ids, protein_input=prot_input_ids, prot_pad=prot_padding_mask, rna_pad=rna_padding_mask)
        mask_rna = (rna_bind != 6) & (rna_bind != 2)
        mask_prot = (prot_bind != 6) & (prot_bind != 2)

        res_rna_valid = res_rna[mask_rna].contiguous() # 强制内存连续
        label_rna_valid = rna_bind[mask_rna].float().contiguous()
        
        res_prot_valid = res_prot[mask_prot].contiguous()
        label_prot_valid = prot_bind[mask_prot].float().contiguous()

        pos_weight = torch.tensor([15.0], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        bce_loss_rna = criterion(res_rna_valid, label_rna_valid.float())
        bce_loss_prot = criterion(res_prot_valid, label_prot_valid.float())

        loss = bce_loss_prot + bce_loss_rna
        # loss = bce_loss_prot + bce_loss_rna + aux_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        res_lis_rna.append(torch.sigmoid(res_rna_valid).detach().cpu())  
        tgt_lis_rna.append(label_rna_valid.detach().cpu())
        res_lis_prot.append(torch.sigmoid(res_prot_valid).detach().cpu())  
        tgt_lis_prot.append(label_prot_valid.detach().cpu())

    res_lis_rna = torch.cat(res_lis_rna).squeeze()  # [N]
    tgt_lis_rna = torch.cat(tgt_lis_rna).squeeze()  # [N]
    res_lis_prot = torch.cat(res_lis_prot).squeeze()  # [N]
    tgt_lis_prot = torch.cat(tgt_lis_prot).squeeze()  # [N]

    # 计算评估指标
    threshold = 0.5  
    pred_labels_rna = (res_lis_rna > threshold).int()  # 根据概率值生成预测标签
    pred_labels_prot = (res_lis_prot > threshold).int()  # 根据概率值生成预测标签
    acc_rna, acc_prot = accuracy_score(tgt_lis_rna, pred_labels_rna), accuracy_score(tgt_lis_prot, pred_labels_prot)  # 准确率
    # auc_rna, auc_prot = roc_auc_score(tgt_lis_rna, res_lis_rna), roc_auc_score(tgt_lis_prot, res_lis_prot)  # AUC
    def safe_roc_auc_score(y_true, y_score):
        """
        鲁棒的 AUC 计算函数
        """
        # 1. 检查 y_score 是否包含 NaN
        if torch.isnan(y_score).any():
            return 0.5  # 或者返回 np.nan，取决于你是否想在日志里看到它
        
        y_true_np = y_true.numpy()
        y_score_np = y_score.numpy()
        
        # 2. 检查是否同时包含两个类别
        if len(np.unique(y_true_np)) < 2:
            return 0.5 # 只有一类数据时，AUC 理论无意义，返回 0.5 占位
            
        return roc_auc_score(y_true_np, y_score_np)

    def safe_auprc_score(y_true, y_score):
        if torch.isnan(y_score).any() or len(np.unique(y_true)) < 2:
            return 0.0 # 或者根据需要返回合理占位符
        return average_precision_score(y_true, y_score)

    # 在 train 和 validate 中替换原有的调用：
    auc_rna = safe_roc_auc_score(tgt_lis_rna, res_lis_rna)
    auc_prot = safe_roc_auc_score(tgt_lis_prot, res_lis_prot)
    mcc_rna, mcc_prot = matthews_corrcoef(tgt_lis_rna, pred_labels_rna), matthews_corrcoef(tgt_lis_prot, pred_labels_prot)  # MCC
    auprc_rna, auprc_prot = safe_auprc_score(tgt_lis_rna, res_lis_rna), safe_auprc_score(tgt_lis_prot, res_lis_prot)  # AUPRC
    
    # 打印整个 epoch 的结果
    avg_loss = total_loss / len(dataloader)
    if local_rank <= 0 and (batch_idx + 1) % 100 == 0:
        # 计算当前阶段的平均 Loss
        current_avg_loss = total_loss / (batch_idx + 1)
        print(f"Step [{batch_idx + 1}/{len(dataloader)}], Loss: {current_avg_loss:.4f}")
    return avg_loss, (acc_rna, acc_prot), (auc_rna, auc_prot), (mcc_rna, mcc_prot), (auprc_rna, auprc_prot)


def validate(model, dataloader, device, local_rank):
    """
    验证模型在验证集上的性能。
    :param model: 模型实例
    :param dataloader: 验证集 DataLoader
    :param device: 设备（如 "cuda" 或 "cpu"）
    :return: 平均损失、ACC、AUC 和 MCC
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    res_lis_rna = []
    res_lis_prot = []
    tgt_lis_rna = []
    tgt_lis_prot = []

    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, batch in enumerate(dataloader):
            # 将数据移动到设备
            prot_input_ids = batch["prot_input_ids"].to(device)
            rna_input_ids = batch["rna_input_ids"].to(device)
            prot_bind = batch["prot_bind"].to(device)
            rna_bind = batch["rna_bind"].to(device)
            prot_padding_mask = batch["prot_padding_mask"].to(device)
            rna_padding_mask = batch["rna_padding_mask"].to(device)

            # 前向传播
            res_rna, res_prot, _ = model(rna_input=rna_input_ids, protein_input=prot_input_ids, prot_pad=prot_padding_mask, rna_pad=rna_padding_mask)
            mask_rna = (rna_bind != 6) & (rna_bind != 2)
            mask_prot = (prot_bind != 6) & (prot_bind != 2)

            res_rna_valid = res_rna[mask_rna].contiguous() # 强制内存连续
            label_rna_valid = rna_bind[mask_rna].float().contiguous()
            
            res_prot_valid = res_prot[mask_prot].contiguous()
            label_prot_valid = prot_bind[mask_prot].float().contiguous()

            # 计算损失
            pos_weight = torch.tensor([15.0], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_rna = criterion(res_rna_valid, label_rna_valid.float())
            loss_prot = criterion(res_prot_valid, label_prot_valid.float())
            loss = loss_rna + loss_prot
            # 累积总损失
            total_loss += loss.item()

            # 收集预测值和真实值
            res_lis_rna.append(torch.sigmoid(res_rna_valid).detach().cpu())  
            tgt_lis_rna.append(label_rna_valid.detach().cpu())
            res_lis_prot.append(torch.sigmoid(res_prot_valid).detach().cpu())  
            tgt_lis_prot.append(label_prot_valid.detach().cpu())

        res_lis_rna = torch.cat(res_lis_rna).squeeze()  # [N]
        tgt_lis_rna = torch.cat(tgt_lis_rna).squeeze()  # [N]
        res_lis_prot = torch.cat(res_lis_prot).squeeze()  # [N]
        tgt_lis_prot = torch.cat(tgt_lis_prot).squeeze()  # [N]
        # 计算评估指标
        threshold = 0.5  # 设置分类阈值
        pred_labels_rna = (res_lis_rna > threshold).int()  # 根据概率值生成预测标签
        pred_labels_prot = (res_lis_prot > threshold).int()  # 根据概率值生成预测标签
        acc_rna, acc_prot = accuracy_score(tgt_lis_rna, pred_labels_rna), accuracy_score(tgt_lis_prot, pred_labels_prot)  # 准确率
        # auc_rna, auc_prot = roc_auc_score(tgt_lis_rna, res_lis_rna), roc_auc_score(tgt_lis_prot, res_lis_prot)  # AUC
        def safe_roc_auc_score(y_true, y_score):
            """
            鲁棒的 AUC 计算函数
            """
            # 1. 检查 y_score 是否包含 NaN
            if torch.isnan(y_score).any():
                return 0.5  # 或者返回 np.nan，取决于你是否想在日志里看到它
            
            y_true_np = y_true.numpy()
            y_score_np = y_score.numpy()
            
            # 2. 检查是否同时包含两个类别
            if len(np.unique(y_true_np)) < 2:
                return 0.5 # 只有一类数据时，AUC 理论无意义，返回 0.5 占位
                
            return roc_auc_score(y_true_np, y_score_np)
        
        def safe_auprc_score(y_true, y_score):
            if torch.isnan(y_score).any() or len(np.unique(y_true)) < 2:
                return 0.0 # 或者根据需要返回合理占位符
            return average_precision_score(y_true, y_score)

        # 在 train 和 validate 中替换原有的调用：
        auc_rna = safe_roc_auc_score(tgt_lis_rna, res_lis_rna)
        auc_prot = safe_roc_auc_score(tgt_lis_prot, res_lis_prot)
        mcc_rna, mcc_prot = matthews_corrcoef(tgt_lis_rna, pred_labels_rna), matthews_corrcoef(tgt_lis_prot, pred_labels_prot)  # MCC
        auprc_rna, auprc_prot = safe_auprc_score(tgt_lis_rna, res_lis_rna), safe_auprc_score(tgt_lis_prot, res_lis_prot)  # AUPRC
        f1_rna, f1_prot = f1_score(tgt_lis_rna, pred_labels_rna), f1_score(tgt_lis_prot, pred_labels_prot)  # F1

    # 打印验证集的结果
    avg_loss = total_loss / len(dataloader)
    if local_rank <= 0:
        print(f"Validation Summary: Avg Loss = {avg_loss:.4f}, "
            f"rna---ACC = {acc_rna:.4f}, AUC = {auc_rna:.4f}, MCC = {mcc_rna:.4f}, AUPRC = {auprc_rna:.4f}, F1 = {f1_rna:.4f}, "
            f"prot---ACC = {acc_prot:.4f}, AUC = {auc_prot:.4f}, MCC = {mcc_prot:.4f}, AUPRC = {auprc_prot:.4f}, F1 = {f1_prot:.4f}")

    return avg_loss, (acc_rna, acc_prot), (auc_rna, auc_prot), (mcc_rna, mcc_prot), (auprc_rna, auprc_prot), (f1_rna, f1_prot)

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
    # parser.add_argument("--local_rank", type=int, default=3, help="Local rank for distributed training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--output_folder", type=str, default="/data/ymxue/p4_protna/code/logs/bindingsite/", help="Output folder for logs.")
    parser.add_argument("--data_folder", type=str, default="/data/ymxue/p4_protna/code/logs/bindingsite/", help="Output folder for logs.")

    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    def set_seed(seed):
        random.seed(seed)  # 设置 Python 的随机种子
        np.random.seed(seed)  # 设置 NumPy 的随机种子
        torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
        torch.cuda.manual_seed(seed)  # 设置 CUDA 的随机种子（单 GPU）
        torch.cuda.manual_seed_all(seed)  # 设置 CUDA 的随机种子（多 GPU）
        torch.backends.cudnn.deterministic = True  # 确保 CuDNN 的卷积操作是确定性的
        torch.backends.cudnn.benchmark = False  # 关闭 CuDNN 的自动优化以保证确定性

    # --- 修改 1: 初始化分布式环境 ---
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", args.local_rank)

    set_seed(42 + args.local_rank if args.local_rank != -1 else 0)

    # 创建输出文件夹
    if args.local_rank <= 0:
        os.makedirs(args.output_folder, exist_ok=True)
    min_train_loss = 100000
    best_epoch = 0

    # --- 修改 2: DataLoader 使用 DistributedSampler ---
    train_dataset = DownstreamDS_token(f"{args.data_folder}train.csv")
    val_dataset = DownstreamDS_token(f"{args.data_folder}test.csv")

    train_sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    
    # 注意：使用 Sampler 时 shuffle 必须为 False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 修改 3: 模型包装为 DDP ---
    model = ModelFineTune(alpha=args.alpha, hidden_dim=64, pretrained_weights_path=args.pretrained_weights_path).to(device)
    
    if args.local_rank != -1:
        # find_unused_parameters=True 视你的 Model 内部是否有未使用的分支而定
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True, gradient_as_bucket_view=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- 修改 4: 训练循环增加 sampler set_epoch (保证每轮 shuffle 不同) ---
    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 只在主进程打印和记录
        is_main_process = (args.local_rank <= 0)
        
        # 传入 local_rank 用于控制打印
        avg_loss, acc, auc, mcc, auprc = train(model, train_loader, optimizer, device=device, local_rank=args.local_rank)
        
        if is_main_process:
            # 验证
            val_model = model.module if hasattr(model, "module") else model
            val_avg_loss, val_acc, val_auc, val_mcc, val_auprc, val_f1 = validate(model, val_loader, device=device, local_rank=args.local_rank)
            
            # 日志写入 (防止 4 个进程同时写同一个文件导致内容错乱)
            with open(f"{args.output_folder}log_train.txt", "a") as f:
                f.write(f"train --- Epoch: {epoch}, loss: {avg_loss}, acc: {acc}, auc: {auc}, mcc: {mcc}, auprc: {auprc}\n")
            with open(f"{args.output_folder}log_val.txt", "a") as f:
                f.write(f"valid --- Epoch: {epoch}, loss: {val_avg_loss}, acc: {val_acc}, auc: {val_auc}, mcc: {val_mcc}, auprc: {val_auprc}, f1: {val_f1}\n")
            
            # 保存逻辑
            if avg_loss < min_train_loss:
                min_train_loss = avg_loss
                best_epoch = epoch
                # 重要：DDP 模型必须通过 .module 获取 state_dict
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), f"{args.output_folder}best_model.pt")
        
        # 每一轮结束后，所有进程同步一次，防止进度不一致
        # if args.local_rank != -1:
        #     dist.barrier(device_ids=[args.local_rank])
