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
        self.head = DownstreamHead(hidden_dim)

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

    def forward(self, protein_input, rna_input=None, prot_pad=None, rna_pad=None):
        # print("forward")
        complex_embed, rna_embed, protein_embed, aux_loss, fake_rna = self.model(rna_input, protein_input, prot_pad, rna_pad, mode='generate')
        # print("forward1")
        res_rna = self.head(rna_embed)
        res_protein = self.head(protein_embed)
        return res_rna, res_protein, fake_rna

# 训练函数
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, average_precision_score, f1_score

def train(model, dataloader, optimizer, device):
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
        res_rna, res_prot, aux_loss = model(rna_input=rna_input_ids, protein_input=prot_input_ids, prot_pad=prot_padding_mask, rna_pad=rna_padding_mask)
        res_rna_valid = res_rna[rna_bind != 6]
        res_prot_valid = res_prot[prot_bind != 6]
        label_rna_valid = rna_bind[rna_bind != 6]
        label_prot_valid = prot_bind[prot_bind != 6]
        
        bce_loss_rna = nn.BCEWithLogitsLoss(reduction='mean')(res_rna_valid, label_rna_valid.float())
        bce_loss_prot = nn.BCEWithLogitsLoss(reduction='mean')(res_prot_valid, label_prot_valid.float())

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
    auc_rna, auc_prot = roc_auc_score(tgt_lis_rna, res_lis_rna), roc_auc_score(tgt_lis_prot, res_lis_prot)  # AUC
    mcc_rna, mcc_prot = matthews_corrcoef(tgt_lis_rna, pred_labels_rna), matthews_corrcoef(tgt_lis_prot, pred_labels_prot)  # MCC

    # 打印整个 epoch 的结果
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch Summary: Avg Loss = {avg_loss:.4f}, "
          f"ACC = {acc_rna:.4f}, AUC = {auc_rna:.4f}, MCC = {mcc_rna:.4f}, "
          f"ACC = {acc_prot:.4f}, AUC = {auc_prot:.4f}, MCC = {mcc_prot:.4f}")

    return avg_loss, (acc_rna, acc_prot), (auc_rna, auc_prot), (mcc_rna, mcc_prot)

def validate(model, dataloader, device):
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
            res_rna, res_prot, aux_loss = model(rna_input=rna_input_ids, protein_input=prot_input_ids, prot_pad=prot_padding_mask, rna_pad=rna_padding_mask)
            res_rna_valid = res_rna[rna_bind != 6]
            res_prot_valid = res_prot[prot_bind != 6]
            label_rna_valid = rna_bind[rna_bind != 6]
            label_prot_valid = prot_bind[prot_bind != 6]

            # 计算损失
            bce_loss_rna = nn.BCEWithLogitsLoss(reduction='mean')(res_rna_valid, label_rna_valid.float())
            bce_loss_prot = nn.BCEWithLogitsLoss(reduction='mean')(res_prot_valid, label_prot_valid.float())
            loss = bce_loss_prot + bce_loss_rna
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
        auc_rna, auc_prot = roc_auc_score(tgt_lis_rna, res_lis_rna), roc_auc_score(tgt_lis_prot, res_lis_prot)  # AUC
        mcc_rna, mcc_prot = matthews_corrcoef(tgt_lis_rna, pred_labels_rna), matthews_corrcoef(tgt_lis_prot, pred_labels_prot)  # MCC
        auprc_rna, auprc_prot = average_precision_score(tgt_lis_rna, res_lis_rna), average_precision_score(tgt_lis_prot, res_lis_prot)  # AUPRC
        f1_rna, f1_prot = f1_score(tgt_lis_rna, pred_labels_rna), f1_score(tgt_lis_prot, pred_labels_prot)  # F1

    # 打印验证集的结果
    avg_loss = total_loss / len(dataloader)
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--output_folder", type=str, default="/data/ymxue/p4_protna/code/logs/bindingsite/", help="Output folder for logs.")
    parser.add_argument("--data_folder", type=str, default="/data/ymxue/p4_protna/code/logs/bindingsite/", help="Output folder for logs.")

    args = parser.parse_args()

    def set_seed(seed):
        random.seed(seed)  # 设置 Python 的随机种子
        np.random.seed(seed)  # 设置 NumPy 的随机种子
        torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
        torch.cuda.manual_seed(seed)  # 设置 CUDA 的随机种子（单 GPU）
        torch.cuda.manual_seed_all(seed)  # 设置 CUDA 的随机种子（多 GPU）
        torch.backends.cudnn.deterministic = True  # 确保 CuDNN 的卷积操作是确定性的
        torch.backends.cudnn.benchmark = False  # 关闭 CuDNN 的自动优化以保证确定性

    # 设置随机种子
    set_seed(42)

    # 创建输出文件夹
    os.makedirs(args.output_folder, exist_ok=True)
    min_train_loss = 100000
    best_epoch = 0

    # 构建 DataLoader
    train_file = f"{args.data_folder}train_filtered.csv"
    val_file = f"{args.data_folder}test_filtered.csv"
    train_dataset = DownstreamDS_token(train_file)  # 预训练模式
    val_dataset = DownstreamDS_token(val_file)  # 预训练模式
                                    
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(len(train_loader), len(val_loader))

    # 初始化模型、优化器和损失函数
    # model = ModelFineTune(vocab_size=args.vocab_size, hidden_dim=args.hidden_dim).to(args.device)
    # model.load_pretrained_weights(args.pretrained_weights_path)
    model = ModelFineTune(alpha=args.alpha, hidden_dim=64, pretrained_weights_path=args.pretrained_weights_path).to(args.device)
    # model = ModelFineTune(vocab_size=vocab_size, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 训练模型
    for epoch in range(args.num_epochs):
        if epoch % 10 == 0:
            print(f"Processing Epoch {epoch}")
        avg_loss, acc, auc, mcc = train(model, train_loader, optimizer, device=args.device)
        val_avg_loss, val_acc, val_auc, val_mcc, val_auprc, val_f1 = validate(model, val_loader, device=args.device)
        with open(f"{args.output_folder}log_train.txt", "a") as f:
            f.write(f"train --- Epoch: {epoch}, loss: {avg_loss}, acc: {acc}, auc: {auc}, mcc: {mcc}\n")
        with open(f"{args.output_folder}log_val.txt", "a") as f:
            f.write(f"valid --- Epoch: {epoch}, loss: {val_avg_loss}, acc: {val_acc}, auc: {val_auc}, mcc: {val_mcc}, auprc: {val_auprc}, f1: {val_f1}\n")
        if avg_loss < min_train_loss:
            min_train_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"{args.output_folder}best_model.pt")
    with open(f"{args.output_folder}log_train.txt", "a") as f:
        f.write(f"Best Epoch: {best_epoch}\n")


