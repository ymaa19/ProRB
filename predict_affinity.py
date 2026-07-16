import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.models.esmc import ESMC
from esm.sdk.api import LogitsConfig, ESMProteinTensor
from model.model import * # 假设 RnaFmModel 在这里定义
from model.attn import ProRComAttnModule

# --- 核心模型定义 (保留结构以支持权重加载) ---

class DownstreamHead(nn.Module):
    def __init__(self, input_dim):
        super(DownstreamHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(32, 1),
        )

    def forward(self, x, padding_mask):
        b, p, r, d = x.shape
        mask = padding_mask.unsqueeze(-1).float()
        masked_x = x * mask
        num_valid_elements = mask.sum(dim=(1, 2), keepdim=True) + 1e-8
        pooled_x = masked_x.sum(dim=(1, 2)) / num_valid_elements.squeeze(-1).squeeze(-1)
        output = self.fc(pooled_x).squeeze(-1)
        return output

class ProteinRnaModel(nn.Module):
    """
    精简后的模型类，仅用于推理
    """
    def __init__(self, hidden_dim=64, alpha=1.0):
        super(ProteinRnaModel, self).__init__()
        # 加载基础模型
        self.rna_model = RnaFmModel.from_pretrained("/path/to/rnafm")
        self.prot_model = ESMC(d_model=960, n_heads=12, n_layers=30, tokenizer=EsmSequenceTokenizer())
        
        self.prot_norm = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, 64),
        )
        self.rna_norm = nn.Sequential(
            nn.Linear(640, 256), nn.ReLU(), nn.LayerNorm(256), nn.Linear(256, 64),
        )
        self.prorcom_attn = ProRComAttnModule(dim=hidden_dim, alpha=alpha)
        self.head = DownstreamHead(hidden_dim)

    def forward(self, rna_id, prot_id, prot_pad, rna_pad):
        # Protein embedding
        prot_tensor = ESMProteinTensor(sequence=prot_id)
        prot_self_emb = self.prot_model.logits(prot_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        prot_self_emb = self.prot_norm(prot_self_emb.logits.sequence.to(torch.float))

        # RNA embedding
        rna_self_emb = self.rna_model(rna_id, rna_pad)
        rna_self_emb = self.rna_norm(rna_self_emb.last_hidden_state.to(torch.float))

        # Attention Mechanism
        complex_emb, rna_embed, protein_embed, _, _ = self.prorcom_attn(rna_self_emb, prot_self_emb, prot_pad, rna_pad)
        
        # Head prediction
        padding_mask = prot_pad.unsqueeze(2) & rna_pad.unsqueeze(1)
        res = self.head(complex_emb, padding_mask)        
        return res

@torch.no_grad()
def run_inference(model_path, data_sample, device="cuda"):
    """
    加载模型并运行单个或批次推理
    """
    # 1. 初始化模型
    model = ProteinRnaModel().to(device)
    
    # 2. 加载训练好的权重
    state_dict = torch.load(model_path, map_location=device)
    # 如果权重中包含 EMA shadow weights，直接加载到 model 中
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. 准备数据
    prot_ids = data_sample["prot_input_ids"].to(device)
    rna_ids = data_sample["rna_input_ids"].to(device)
    prot_mask = data_sample["prot_padding_mask"].to(device)
    rna_mask = data_sample["rna_padding_mask"].to(device)

    # 4. 前向传播
    prediction = model(rna_ids, prot_ids, prot_mask, rna_mask)
    
    return prediction.cpu().numpy()

if __name__ == "__main__":
    # 使用示例
    MODEL_WEIGHTS = "outputs/fold_4_best_model.pt"
    # 这里假设你已经有了预处理好的 batch 数据
    # result = run_inference(MODEL_WEIGHTS, sample_batch)
    # print(f"Predicted affinity: {result}")
    print("Inference script ready.")
