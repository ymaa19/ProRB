import torch
import torch.nn as nn
import torch.nn.functional as F

class RowWiseAttention(nn.Module):
    """
    renew rna attention
    """
    def __init__(self, dim, num_heads=8):
        super(RowWiseAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, padding_mask=None):
        B, P, R, D = x.shape  # B: batch size, P: protein length, R: residue length, D: feature dimension
        qkv = self.qkv(x).reshape(B, P, R, 3, self.num_heads, D // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, H, P, R, R)
        if padding_mask is not None: # padding_mask : (B, P, R)
            padding_mask = padding_mask.bool() # (B, P, R)
            attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(4), float('-inf'))
        attn = F.softmax(attn, dim=-1) 

        out = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B, P, R, D)
        out = self.proj(out) # (B, P, R, D)
        return out 

class ColumnWiseAttention(nn.Module):
    """
    renew protein attention
    """
    def __init__(self, dim, num_heads=8):
        super(ColumnWiseAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, padding_mask=None):
        B, P, R, D = x.shape  # B: batch size, P: protein length, R: residue length, D: feature dimension
        qkv = self.qkv(x).reshape(B, P, R, 3, self.num_heads, D // self.num_heads).permute(3, 0, 4, 2, 1, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, R, P, D//H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, R, P, P)
        if padding_mask is not None:
            padding_mask = padding_mask.bool() # (B, P, R)
            padding_mask = padding_mask.permute(0, 2, 1)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(4) # (B, 1, R, 1, P)
            attn = attn.masked_fill(padding_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B, P, R, D)  # (B, P, R, D)
        return out, attn # (B, P, R, D)
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, padding_mask=None):
        B, N_q, D = query.shape
        _, N_k, _ = key.shape
        if padding_mask is not None:
            _, len1, len2 = padding_mask.shape
            if len1 == N_k and len2 == N_q:
                padding_mask = padding_mask.permute(0, 2, 1)

        q = self.query_proj(query).reshape(B, N_q, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N_q, D/H)
        k = self.key_proj(key).reshape(B, N_k, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)      # (B, H, N_k, D/H)
        v = self.value_proj(value).reshape(B, N_k, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N_k, D/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_k)
        if padding_mask is not None:
            inverted_padding_mask = ~padding_mask
            attn = attn.masked_fill(inverted_padding_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, D)  # (B, N_q, D)
        out = self.out_proj(out)
        return out, attn

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, padding_mask=None):
        B, N, D = x.shape  # B: batch size, N: sequence length, D: feature dimension
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if padding_mask is not None: # padding_mask: (B, N)
            inverted_padding_mask = ~padding_mask
            attn = attn.masked_fill(inverted_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(out)
        return out, attn
