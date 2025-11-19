import torch.nn as nn
import torch
from model.attn import SelfAttention, CrossAttention
import torch.nn.functional as F

def generate_per_sample_masks(sz, device):
    # Create a boolean causal mask where True indicates positions to be masked
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask

def make_key_padding_mask(lengths):
    bsz = lengths.size(0)
    # Construct a [bsz, max_len] matrix indicating padding positions
    key_padding_mask = torch.arange(512, device=lengths.device).expand(bsz, -1) >= lengths.unsqueeze(1)
    return key_padding_mask.bool()  # Already bool, no change needed

class FakeRNADecoder(nn.Module):
    def __init__(self, dim=64, num_heads=4, num_layers=2, max_rna_length=512):
        super(FakeRNADecoder, self).__init__()
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Positional Encoding for RNA (optional)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_rna_length, dim))
        self.start_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Linear projection to match dimensions (if needed)
        self.protein_proj = nn.Linear(dim, dim)
        self.rna_proj = nn.Linear(dim, dim)
        self.max_rna_length = max_rna_length

    def forward(self, protein_features, rna_features=None, rna_lengths=None):
        """
        Args:
            protein_features: Tensor of shape (bsz, P, dim), representing protein embeddings.
            rna_features: Tensor of shape (bsz, R, dim), representing real RNA embeddings (optional).
            rna_length: Tensor of shape (bsz,), the target length of fake RNA sequence.

        Returns:
            fake_rna: Tensor of shape (bsz, R, dim), representing generated fake RNA embeddings.
        """
        bsz, _, _ = protein_features.shape

        # Step 1: Project protein features
        protein_features = self.protein_proj(protein_features)  # (bsz, P, dim)

        # Step 2: Generate initial RNA tokens
        if rna_features is not None:
            tgt = self.rna_proj(rna_features)
            tgt = torch.cat([self.start_token.expand(bsz, -1, -1), tgt[:, :-1]], dim=1)
            R = rna_features.size(1)
        else:
            if rna_lengths is None:
                R = self.max_rna_length
                rna_lengths = torch.full((bsz,), R, device=protein_features.device)
            else:
                R = rna_lengths.max().item()
            
            # Combine start token and positional encoding
            start_tokens = self.start_token.expand(bsz, -1, -1)
            pos_enc = self.positional_encoding[:, :R-1, :]
            tgt = torch.cat([start_tokens, pos_enc], dim=1)
            tgt = self.rna_proj(tgt)
        
        # Step 3: Generate masks
        tgt_mask = generate_per_sample_masks(R, protein_features.device)
        tgt_key_padding_mask = make_key_padding_mask(rna_lengths)
        
        # Step 4: Transformer decoding
        fake_rna = self.transformer_decoder(
            tgt=tgt,
            memory=protein_features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return fake_rna

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # Learnable scaling parameter

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale
    
class ProRComAttnModule(nn.Module):
    def __init__(self, dim, num_heads=4, alpha=1.0):
        super(ProRComAttnModule, self).__init__()
        self.self_attention_rna = SelfAttention(dim, num_heads)
        self.self_attention_protein = SelfAttention(dim, num_heads)
        self.cross_attention_rna_to_protein = CrossAttention(dim, num_heads)
        self.cross_attention_protein_to_rna = CrossAttention(dim, num_heads)
        self.rna_decoder = FakeRNADecoder()

        self.temperature = 0.8
        self.alpha = alpha

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        self.rnanorm_after_self_attn = RMSNorm(dim)
        self.protnorm_after_self_attn = RMSNorm(dim)
        self.rnanorm_after_cross_attn = RMSNorm(dim)
        self.protnorm_after_cross_attn = RMSNorm(dim)
        self.rna_norm = RMSNorm(dim)
        self.protein_norm = RMSNorm(dim)

    def forward(self, rna_features, protein_features, prot_pad=None, rna_pad=None):
        protein_mask = prot_pad if prot_pad is not None else None
        rna_mask = rna_pad if rna_pad is not None else None

        # print(rna_valid_len.shape)

        # Step 1: Self Attention
        if rna_features is None:
            # 微调阶段：如果没有 RNA 输入，则直接使用解码器生成 fake RNA
            fake_rna = self.rna_decoder(protein_features)  # 传入 None 表示没有 RNA 输入
            rna_features = fake_rna  # 使用生成的 fake RNA 替代真实 RNA
            aux_loss_rna = torch.tensor(0.0, device=protein_features.device)  # 不计算辅助损失
        else:
            rna_valid_len = torch.sum(rna_pad, dim=-1)
            alpha = self.alpha
            fake_rna = self.rna_decoder(protein_features=protein_features, rna_features=rna_features, rna_lengths=rna_valid_len)
            real_rna_soft = F.log_softmax(rna_features / self.temperature, dim=-1)
            fake_rna_soft = F.softmax(fake_rna / self.temperature, dim=-1)
            aux_loss_rna = self.kl_loss(real_rna_soft, fake_rna_soft) * (self.temperature ** 2)
            rna_features = alpha * rna_features + (1 - alpha) * fake_rna
        # Step 0.1: Check if both modal exist
        rna_features = self.rna_norm(rna_features)
        protein_features = self.protein_norm(protein_features)

        # Step 1: Self-Attention for local context modeling
        # print("去掉 self attention")
        rna_features = self.self_attention_rna(rna_features, rna_mask)[0] + rna_features
        rna_feat_aft_selfattn = self.self_attention_rna(rna_features, rna_mask)[1]
        rna_features = self.rnanorm_after_self_attn(rna_features)
        protein_features = self.self_attention_protein(protein_features, protein_mask)[0] + protein_features
        protein_features = self.protnorm_after_self_attn(protein_features)

        # print("self attention done")
        # Step 2: Cross-Attention for interaction modeling
        rna_to_protein, attn1 = self.cross_attention_rna_to_protein(rna_features, protein_features, protein_features) # (B, R, dim)
        protein_to_rna, attn2 = self.cross_attention_protein_to_rna(protein_features, rna_features, rna_features) # (B, P, dim)
        protein_features_renewed = protein_features + protein_to_rna
        protein_features_renewed = self.protnorm_after_cross_attn(protein_features_renewed)
        rna_features_renewed = rna_features + rna_to_protein
        rna_features_renewed = self.rnanorm_after_cross_attn(rna_features_renewed)
        aux_loss = aux_loss_rna
        complex_features = torch.einsum('bpd,brd->bprd', protein_features_renewed, rna_features_renewed)  # (B, P, R, dim)
        return complex_features, rna_features_renewed, protein_features_renewed, aux_loss, (fake_rna, rna_feat_aft_selfattn, attn1, attn2)
        
        # ### 替换去除crossattn
        # aux_loss = aux_loss_rna
        # complex_features = torch.einsum('bpd,brd->bprd', protein_features, rna_features)  # (B, P, R, dim)
        # return complex_features, rna_features, protein_features, aux_loss, fake_rna


class ProRComAttn(nn.Module):
    def __init__(self, dim, alpha=1.0):
        super(ProRComAttn, self).__init__()
        self.layers = nn.ModuleList([ProRComAttnModule(dim, alpha=alpha) for _ in range(1)])
        
    # 定义前向传播函数
    def forward(self, rna_features, protein_features, prot_pad=None, rna_pad=None):
        # 遍历每一层
        for i, layer in enumerate(self.layers):
            # 调用每一层的forward函数，传入rna_features、protein_features、prot_pad、rna_pad参数
            complex_features, rna_features, protein_features, aux_loss, _ = layer(rna_features, protein_features, prot_pad=prot_pad, rna_pad=rna_pad)
        # 返回complex_features、rna_features、protein_features、aux_loss
        return complex_features, rna_features, protein_features, aux_loss