from utils.dataset import *
from utils.tokenizer import *
from model.model import *
from model.attn import *
from esm.models.esmc import ESMC
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.sdk.api import LogitsConfig, ESMProteinTensor
from multimolecule import RnaFmModel, RnaFmConfig
class Model(nn.Module):
    def __init__(self, hidden_dim=64, alpha=0.4):
        super(Model, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rna_config = RnaFmConfig.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        # 强制设回旧版参数（根据你记忆或文档）
        self.rna_config.emb_layer_norm_after = False   # 示例
        self.rna_config.vocab_size = 26 
        self.rna_model = RnaFmModel.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm", config=self.rna_config, add_pooling_layer=False)

         # Load ESMC model
         
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
        self.rna_norm_gen = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
        )
        self.alpha = alpha
        self.prorcom_attn = ProRComAttnModule(dim=hidden_dim, alpha=self.alpha)
        self.token_emb = nn.Embedding(26, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 512 + 1, hidden_dim)) # 预留位置编码
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True
        )

        for param in self.rna_model.parameters():
            param.requires_grad = False
        for param in self.prot_model.parameters():
            param.requires_grad = False

    def forward(self, rna_id, prot_id, prot_pad=None, rna_pad=None, mode="train"):
        # print("forward")
        # prot_tensor = ESMProteinTensor()
        # prot_tensor.sequence = prot_id.to(self.prot_model.device)
        prot_id = prot_id.to(device=self.prot_model.device, dtype=torch.long)
        batched_prot_tensor = _BatchedESMProteinTensor(sequence=prot_id)
        prot_self_emb = self.prot_model.logits(batched_prot_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        prot_self_emb = self.prot_norm(prot_self_emb.logits.sequence.to(torch.float))
        # rna
        if mode != "generate":
            rna_self_emb = self.rna_model(rna_id, rna_pad) if rna_id is not None else None
            rna_self_emb = self.rna_norm(rna_self_emb.last_hidden_state.to(torch.float)) if rna_self_emb is not None else None
        else:
            seq_len = rna_id.size(1)
            x = self.token_emb(rna_id) + self.pos_emb[:, :seq_len, :]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=rna_id.device), diagonal=1).bool()
            rna_self_emb = self.encoder_layer(x, src_mask=causal_mask, src_key_padding_mask=~(rna_pad.bool()))
            rna_self_emb = self.rna_norm_gen(rna_self_emb.to(torch.float))
        # print('rna_self_emb shape:', rna_self_emb.shape)
        complex_emb, rna_embed, protein_embed, aux_loss, fake_rna_combo = self.prorcom_attn(rna_self_emb, prot_self_emb, prot_pad, rna_pad)
        fake_rna = fake_rna_combo[-1]
        return complex_emb, rna_embed, protein_embed, aux_loss, fake_rna

class proj_token(nn.Module):
    def __init__(self, hidden_dim, alpha):
        super(proj_token, self).__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.prot_proj = nn.Linear(hidden_dim, 33)
        self.rna_proj = nn.Linear(hidden_dim, 26)

        self.rna_decode = nn.Linear(hidden_dim, 26)

        self.alpha = alpha

        self.model = Model(hidden_dim=64, alpha=self.alpha)

    def forward(self, rna_id, prot_id, prot_pad, rna_pad):
        _, rna_emb, protein_emb, aux_loss, fake_rna = self.model(rna_id, prot_id, prot_pad=prot_pad, rna_pad=rna_pad, mode='train')
        rna_emb = self.rna_proj(rna_emb)
        protein_emb = self.prot_proj(protein_emb)

        rna_decode = self.rna_decode(fake_rna)
        return rna_emb, protein_emb, aux_loss, rna_decode

# 训练函数
def train(model, dataloader, optimizer, criterion, device, log_file):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移动到设备
        prot_input_ids = batch["prot_input_ids"].to(device)
        prot_labels = batch["prot_labels"].to(device)
        prot_padding_mask = batch["prot_padding_mask"].to(device)
        rna_input_ids = batch["rna_input_ids"].to(device)
        rna_labels = batch["rna_labels"].to(device)
        rna_padding_mask = batch["rna_padding_mask"].to(device)        

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        rna_embed, protein_embed, aux_loss, rna_decode = model(rna_input_ids, prot_input_ids, prot_pad=prot_padding_mask, rna_pad=rna_padding_mask)

        loss_prot = criterion(protein_embed.view(-1, 33), prot_labels.view(-1))
        loss_rna = criterion(rna_embed.view(-1, 26), rna_labels.view(-1))

        loss_extra = criterion(rna_decode.view(-1, 26), rna_labels.view(-1))

        # loss = loss_prot + loss_rna + aux_loss + loss_extra
        loss = loss_prot + loss_rna + aux_loss

        # 打印损失
        if batch_idx % 500 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item()} loss_prot = {loss_prot.item()} loss_rna = {loss_rna.item()} aux_loss = {aux_loss.item()} loss_extra={loss_extra.item()}")
            #######################################################################################
            #######################################################################################
            #######################################################################################
            #######################################################################################
            with open(log_file, "a") as f:
                f.write(f"Batch {batch_idx}: Loss = {loss.item()} loss_prot = {loss_prot.item()} loss_rna = {loss_rna.item()} aux_loss = {aux_loss.item()} loss_extra={loss_extra.item()}\n")
            #######################################################################################
            #######################################################################################
            #######################################################################################
        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

import argparse
import os
# 主程序
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a protein-RNA interaction model.")

    # 添加命令行参数
    parser.add_argument("--vocab_size", type=int, default=64, help="Vocabulary size for the model.")
    parser.add_argument("--max_prot_len", type=int, default=1024, help="Maximum length of protein sequences.")
    parser.add_argument("--max_rna_len", type=int, default=512, help="Maximum length of RNA sequences.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use for training (e.g., cuda:2).")
    parser.add_argument("--log_file", type=str, default="training_log.txt", help="File to save training logs.")
    parser.add_argument("--model_save_path", type=str, default="pretrained_model.pth", help="Path to save the trained model.")
    parser.add_argument("--folder", type=str, default="/data/ymxue/p4_protna/data/CoPRA/cleaned_data/pretrain.csv", help="Path to the pretrain data.")
    # 解析命令行参数
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    # tokenizer = CustomTokenizer()
    # tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})
    # 数据集路径
    csv_file = "/data/ymxue/p4_protna/data/CoPRA/cleaned_data/pretrain.csv"
    # 构建 DataLoader
    dataset = ProteinRNADataset(csv_file, 
                                max_prot_len=args.max_prot_len,
                                max_rna_len=args.max_rna_len)  
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 构建 DataLoader
    # 初始化模型、优化器和损失函数
    # model = Model(alpha=args.alpha).to(args.device)
    model = proj_token(hidden_dim=64, alpha=args.alpha).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充位置
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # print(model)
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
