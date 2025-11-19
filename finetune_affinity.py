import sys
sys.path.append("..")

from utils.dataset import *
from utils.tokenizer import *
from model.model import *
from model.attn import *

# 下游任务头
class DownstreamHead(nn.Module):
    def __init__(self, input_dim):
        super(DownstreamHead, self).__init__()
        # need dropout
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(32, 1),
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, padding_mask):
        b, p, r, d = x.shape
        mask = padding_mask.unsqueeze(-1).float()  # [b, p, r, 1]
        masked_x = x * mask  # [b, p, r, d]
        num_valid_elements = mask.sum(dim=(1, 2), keepdim=True) + 1e-8  # [b, 1, 1, 1]
        pooled_x = masked_x.sum(dim=(1, 2)) / num_valid_elements.squeeze(-1).squeeze(-1)  # [b, d]
        output = self.fc(pooled_x).squeeze(-1)  # [b]
        # print("output:" , output)
        return output

from esm.models.esmc import ESMC
from esm.sdk.api import LogitsConfig, ESMProteinTensor

class Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, alpha=1.0):
        super(Model, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rna_model = RnaFmModel.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        self.prot_model = ESMC(d_model=960, n_heads=12, n_layers=30, tokenizer=EsmSequenceTokenizer())
        self.prot_model.load_state_dict(torch.load("/data/ymxue/p4_protna/code/fm_model/esm3/data/weights/esmc_300m_2024_12_v0.pth"))

        self.prot_norm = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
        )
        self.rna_norm = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
        )

        self.alpha = alpha
        self.prorcom_attn = ProRComAttnModule(dim=hidden_dim, alpha=alpha)

        for param in self.rna_model.parameters():
            param.requires_grad = False
        for param in self.prot_model.parameters():
            param.requires_grad = False

    def forward(self, rna_id, prot_id, prot_pad=None, rna_pad=None):
        # prot
        prot_tensor = ESMProteinTensor()
        prot_tensor.sequence = prot_id
        prot_self_emb = self.prot_model.logits(prot_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        prot_self_emb = self.prot_norm(prot_self_emb.logits.sequence.to(torch.float))

        # rna
        rna_self_emb = self.rna_model(rna_id, rna_pad) if rna_id is not None else None
        rna_self_emb = self.rna_norm(rna_self_emb.last_hidden_state.to(torch.float)) if rna_self_emb is not None else None

        complex_emb, rna_embed, protein_embed, aux_loss, _ = self.prorcom_attn(rna_self_emb, prot_self_emb, prot_pad, rna_pad)
        return complex_emb, rna_embed, protein_embed, aux_loss

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # 只复制需要更新的参数（requires_grad=True）
        self.shadow = {
            name: param.data.clone().to(param.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.original = {
            name: param.data.clone().to(param.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].copy_(
                        (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    )

    def apply_shadow(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.data.copy_(self.shadow[name])

    def restore_original(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.original:
                    param.data.copy_(self.original[name])


# 微调模型
class ModelFineTune(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, pretrained_weights_path=None, alpha=1.0, device="cuda:0"):
        super(ModelFineTune, self).__init__()
        self.model = Model(hidden_dim, alpha=alpha)
        self.head = DownstreamHead(hidden_dim)

        #############################################################
        self.to(device)

        # EMA 相关变量
        self.use_ema = False
        self.ema_initialized = False
        self.ema = None
        self.ema_decay = 0.995

        #############################################################

        # 加载预训练权重
        if pretrained_weights_path is not None:
            self.load_pretrained_weights(pretrained_weights_path)

        for param in self.model.prorcom_attn.self_attention_protein.parameters():
            param.requires_grad = False
        for param in self.model.prorcom_attn.self_attention_rna.parameters():
            param.requires_grad = False
  
    #############################################################
    def enable_ema(self):
        """启用 EMA，在第一次调用时初始化"""
        self.use_ema = True

    def disable_ema(self):
        self.use_ema = False

    def initialize_ema(self):
        """在训练开始后初始化 EMA（避免初始参数污染）"""
        if not self.ema_initialized:
            self.ema = EMA(self, decay=self.ema_decay)
            self.ema_initialized = True

    def update_ema(self):
        """更新 EMA shadow weights"""
        if self.use_ema and self.ema_initialized:
            self.ema.update(self)
    #############################################################

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
            ####################
            # for param in self.model.parameters():
            #     param.requires_grad = False
            # for param in self.model.prorcom_attn.cross_attention_rna_to_protein.parameters():
            #     param.requires_grad = True
            # for param in self.model.prorcom_attn.cross_attention_protein_to_rna.parameters():
            #     param.requires_grad = True
            # print("Set model parameters to not require gradients.")
            ####################
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    def forward(self, rna_input, protein_input, prot_pad=None, rna_pad=None):
        # 如果处于 eval 模式且启用了 EMA，则应用 shadow weights
        if self.use_ema and self.ema_initialized and not self.training:
            self.ema.apply_shadow(self)
        else:
            pass  # 使用原始参数

        complex_emb, rna_embed, protein_embed, aux_loss = self.model(rna_input, protein_input, prot_pad=prot_pad, rna_pad=rna_pad)
        padding_mask = prot_pad.unsqueeze(2) & rna_pad.unsqueeze(1)
        res = self.head(complex_emb, padding_mask)

        # 计算 cosine similarity 并调整输出
        rna_embed = torch.mean(rna_embed, dim=1)
        protein_embed = torch.mean(protein_embed, dim=1)
        cosine_sim = F.cosine_similarity(rna_embed, protein_embed, dim=-1).mean()
        res = res * (1 - cosine_sim) ** 0.5

        return res, cosine_sim, aux_loss


# 训练函数
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

def calculate_correlations(preds, targets):
    preds = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    pearson_corr, _ = pearsonr(preds, targets)
    spearman_corr, _ = spearmanr(preds, targets)
    return pearson_corr, spearman_corr


def train(model, dataloader, optimizer, scheduler, device, use_ema=True):
    model.train()
    total_loss = 0
    res_lis = []
    tgt_lis = []

    ema_initialized = False  # 控制是否已初始化 EMA

    for batch_idx, batch in enumerate(dataloader):
        prot_input_ids = batch["prot_input_ids"].to(device)
        rna_input_ids = batch["rna_input_ids"].to(device)
        prot_pad = batch["prot_padding_mask"].to(device)
        rna_pad = batch["rna_padding_mask"].to(device)
        value = batch["value"].to(device)

        optimizer.zero_grad()
        res, cosine_sim, aux_loss = model(rna_input_ids, prot_input_ids, prot_pad=prot_pad, rna_pad=rna_pad)
        rmse_loss = torch.sqrt(F.mse_loss(res, value))
        # mae_loss = F.l1_loss(res, value)
        mae_loss_fn = nn.SmoothL1Loss(beta=.5)
        mae_loss = mae_loss_fn(res, value)
        loss = rmse_loss + mae_loss + aux_loss
        # loss = rmse_loss + mae_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        if use_ema and model.use_ema and not ema_initialized:
            model.initialize_ema()
            ema_initialized = True

        if use_ema and model.use_ema and ema_initialized:
            model.update_ema()

        # 日志记录...
        res_lis.append(res.detach())
        tgt_lis.append(value)
        total_loss += loss.item()

    res_lis = torch.cat(res_lis).squeeze()  
    tgt_lis = torch.cat(tgt_lis).squeeze()  

    pearson_corr, spearman_corr = calculate_correlations(res_lis, tgt_lis)
    return loss, rmse_loss, mae_loss, pearson_corr, spearman_corr

def validate(model, dataloader, device):
    """
    验证模型在验证集上的性能。
    :param model: 模型实例
    :param dataloader: 验证集 DataLoader
    :param device: 设备（如 "cuda" 或 "cpu"）
    :return: 平均损失、Pearson 相关系数和 Spearman 相关系数
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    res_lis = []
    tgt_lis = []

    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, batch in enumerate(dataloader):
            # 将数据移动到设备
            prot_input_ids = batch["prot_input_ids"].to(device)
            rna_input_ids = batch["rna_input_ids"].to(device)
            prot_pad = batch["prot_padding_mask"].to(device)
            rna_pad = batch["rna_padding_mask"].to(device)
            value = batch["value"].to(device)

            # 前向传播
            res, cosine_sim, aux_loss = model(rna_input_ids, prot_input_ids, prot_pad=prot_pad, rna_pad=rna_pad)
            # res = res + 1
            res_lis.append(res.detach())
            tgt_lis.append(value)

            # 计算损失
            rmse_loss = torch.sqrt(F.mse_loss(res, value))
            mae_loss = F.l1_loss(res, value)
            # loss = rmse_loss + mae_loss
            loss = rmse_loss + mae_loss + aux_loss

            # 累积总损失
            total_loss += loss.item()

    # 将所有预测值和真实值拼接成一维张量
    res_lis = torch.cat(res_lis).squeeze()  # [N]
    tgt_lis = torch.cat(tgt_lis).squeeze()  # [N]

    # 计算 Pearson 和 Spearman 相关系数
    pearson_corr, spearman_corr = calculate_correlations(res_lis, tgt_lis)

    # 打印验证集的结果
    # avg_loss = total_loss / len(dataloader)
    # print(f"Validation Summary: Avg Loss = {avg_loss:.4f}, "
    #       f"Pearson Corr = {pearson_corr:.4f}, Spearman Corr = {spearman_corr:.4f}")

    return loss, rmse_loss, mae_loss, pearson_corr, spearman_corr

from tqdm import tqdm
import os
import argparse
import numpy as np
# 主程序
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Train and validate a fine-tuned model.")

    # 数据相关参数
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for logs.")
    parser.add_argument("--max_prot_len", type=int, default=1024, help="Maximum length of protein sequences.")
    parser.add_argument("--max_rna_len", type=int, default=512, help="Maximum length of RNA sequences.")

    # 模型相关参数
    parser.add_argument("--vocab_size", type=int, default=64, help="Vocabulary size.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of the model.")
    parser.add_argument("--pretrained_weights_path", type=str, default=None, help="Path to pretrained weights.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for ProRComAttnModule.")

    # 训练相关参数
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use (e.g., 'cuda:0' or 'cpu').")

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
    for i in tqdm(range(4, 5)):
        print(f"Fold {i} training")
        min_train_loss = 100000
        min_val_loss = 100000

        # 构建 DataLoader
        train_file = f"/data/ymxue/p4_protna/code/task1_bind_affinity/exp_cdhit/data/prot_cd_rna_no/train_split_{i}.csv"
        val_file = f"/data/ymxue/p4_protna/code/task1_bind_affinity/exp_cdhit/data/prot_cd_rna_no/test_split_{i}.csv"
        train_dataset = DownstreamDataset(train_file, max_prot_len=1024, max_rna_len=512, mode="pretrain")  # 预训练模式
        val_dataset = DownstreamDataset(val_file, max_prot_len=1024, max_rna_len=512, mode="pretrain")  # 预训练模式
                                        
        train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(len(train_loader), len(val_loader))
        # 初始化模型、优化器和损失函数
        # model = ModelFineTune(vocab_size=args.vocab_size, hidden_dim=args.hidden_dim).to(args.device)
        model = ModelFineTune(vocab_size=args.vocab_size, hidden_dim=args.hidden_dim, pretrained_weights_path=args.pretrained_weights_path, alpha=args.alpha, device=args.device).to(args.device)
        # model = ModelFineTune(vocab_size=vocab_size, hidden_dim=64, pretrained_weights_path="/data/ymxue/p4_protna/pretrained_model.pth").to(device)
        # model = ModelFineTune(vocab_size=vocab_size, hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            total_steps=args.num_epochs * len(train_loader),
            pct_start=0.3,
            anneal_strategy='linear',
            cycle_momentum=True, 
        )

        early_stop_epoch = 0
        best_epoch = 0
        patience = 20
        no_improve = 0
        # 训练模型
        for epoch in range(args.num_epochs):
            if epoch % 10 == 0:
                print(f"Processing Fold {i} Epoch {epoch}")
            if epoch == 3:
                model.initialize_ema()
            use_ema = (epoch >= 3)
            loss, rmse_loss, mae_loss, pearson_corr, spearman_corr = train(model, train_loader, optimizer, scheduler, device=args.device, use_ema=use_ema)
            # model.enable_ema()  # 启用 EMA
            val_loss, val_rmse_loss, val_mae_loss, val_pearson_corr, val_spearman_corr = validate(model, val_loader, device=args.device)
            if val_rmse_loss + val_mae_loss < min_val_loss:
                min_val_loss = val_rmse_loss + val_mae_loss
                early_stop_epoch = epoch
                no_improve = 0
            else: 
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            with open(f"{args.output_folder}fold_{i}_log_train.txt", "a") as f:
                f.write(f"train --- Epoch: {epoch}, loss: {loss}, rmse_loss: {rmse_loss}, mae_loss: {mae_loss}, Pearson Correlation: {pearson_corr}, Spearman Correlation: {spearman_corr}\n")
            with open(f"{args.output_folder}fold_{i}_log_val.txt", "a") as f:
                f.write(f"valid --- Epoch: {epoch}, val_loss: {val_loss}, rmse_loss: {val_rmse_loss}, mae_loss: {val_mae_loss}, Pearson Correlation: {val_pearson_corr}, Spearman Correlation: {val_spearman_corr}\n")
            avg_loss = rmse_loss + mae_loss
            if avg_loss < min_train_loss and epoch > 5:
                min_train_loss = avg_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f"{args.output_folder}fold_{i}_best_model.pt")
        with open(f"{args.output_folder}fold_{i}_log_val.txt", "a") as f:
            f.write(f"Best Epoch: {best_epoch}\n")
        del model
        torch.cuda.empty_cache()


