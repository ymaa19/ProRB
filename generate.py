from esm.sdk.api import ESMProteinTensor
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from esm.utils.sampling import _BatchedESMProteinTensor
import torch

def id_seq_to_string(id_seq):
    """6:A, 7:C, 8:G, 9:U, else:N"""
    id_to_base = {6: 'A', 7: 'C', 8: 'G', 9: 'U'}
    seq = ''.join([id_to_base.get(i, 'N') for i in id_seq])
    return seq

test_file = pd.read_csv("/data/ymxue/p4_protna/code/A_review/t3_gen/d2_1_gen_test.csv")

# --- 定义温度列表 ---
temperatures = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

for temp in temperatures:
    print(f"Starting generation for temperature: {temp}")
    df_all = []
    
    # 这里的循环保持不变
    for idx in tqdm.tqdm(range(len(test_file))):
        prot_seq = test_file.iloc[idx]['prot']

        topk = 4
        # 使用当前循环的温度
        temperature = temp 
        bsz = 256
        valid_tokens = [6, 7, 8, 9]
        eos_token_id = 2

        with torch.no_grad(): 
            prot_input_ids = process_sequence(prot_seq, 1024, prot_tokenizer, mode="prot")
            prot_padding_mask = torch.tensor(
                generate_padding_mask(prot_input_ids, prot_tokenizer.pad_token_id), dtype=torch.bool
            )
            prot_input_ids = prot_input_ids.to(device).unsqueeze(0)
            batched_prot_tensor = _BatchedESMProteinTensor(sequence=prot_input_ids)
            prot_padding_mask = prot_padding_mask.to(device).unsqueeze(0)

            prot_tensor = ESMProteinTensor()
            prot_tensor.sequence = prot_input_ids
            prot_self_emb = prot_model.logits(batched_prot_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            prot_self_emb = prot_norm(prot_self_emb.logits.sequence.to(torch.float))
            prot_self_emb = prot_self_emb.expand(bsz, -1, -1)
            prot_emb = decoder.protein_proj(prot_self_emb)
            
            max_rna_len = 512
            start_token = decoder.start_token
            tgt = start_token.expand(prot_self_emb.size(0), 1, -1).to(device)

            # 初始化状态
            active = torch.ones(prot_self_emb.size(0), dtype=torch.bool, device=device)
            seqs_tokens = [[] for _ in range(prot_self_emb.size(0))]
            cum_log_probs = torch.zeros(prot_self_emb.size(0), device=device)
            lengths = torch.zeros(prot_self_emb.size(0), dtype=torch.long, device=device)

            for t in range(max_rna_len):
                # --- 步骤 1: 整体处理已生成的 token 序列 ---
                if not seqs_tokens[0]:
                    rna_self_emb = None
                else:
                    curr_ids = torch.tensor(seqs_tokens, device=device) 
                    curr_L_tokens = curr_ids.size(1)
                    x = token_emb(curr_ids) + pos_emb[:, :curr_L_tokens, :]
                    causal_mask = torch.triu(torch.ones(curr_L_tokens, curr_L_tokens, device=device), diagonal=1).bool()
                    rna_self_emb = encoder(x, src_mask=causal_mask)
                    rna_self_emb = rna_norm_gen(rna_self_emb.to(torch.float))

                # --- 步骤 2: 构建 tgt ---
                if rna_self_emb is None:
                    tgt = start_token.expand(bsz, -1, -1)
                else:
                    tgt_projected = decoder.rna_proj(rna_self_emb)
                    tgt = torch.cat([start_token.expand(bsz, -1, -1), tgt_projected], dim=1)

                # --- 步骤 3: Decoder 推理 ---
                curr_L = tgt.size(1)
                tgt_mask = torch.triu(torch.ones(curr_L, curr_L, device=device), diagonal=1).bool()
                tgt_key_padding_mask = torch.zeros(bsz, curr_L, dtype=torch.bool, device=device)
                
                decoder_out = decoder.transformer_decoder(
                    tgt=tgt,
                    memory=prot_emb,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )

                # --- 步骤 5: 采样预测下一个 Token ---
                next_rep = decoder_out[:, -1:, :]
                logits = head(next_rep)

                mask = torch.full_like(logits, float('-inf'))
                mask[:, :, valid_tokens] = 0
                mask[:, :, [eos_token_id]] = 0
                logits = logits + mask

                probs = F.softmax(logits / temperature, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
                
                sampled_indices = torch.multinomial(topk_probs.squeeze(1), num_samples=1)
                decode_token_id = torch.gather(topk_indices.squeeze(1), 1, sampled_indices)
                
                token_cpu = decode_token_id.squeeze(1).cpu().numpy()
                log_probs_step = F.log_softmax(logits.squeeze(1), dim=-1)

                for i in range(bsz):
                    raw_token = token_cpu[i]
                    if not active[i]:
                        seqs_tokens[i].append(0) 
                        continue
                    if raw_token == eos_token_id:
                        active[i] = False
                        seqs_tokens[i].append(0)
                        continue
                    seqs_tokens[i].append(raw_token)
                    lengths[i] += 1
                    cum_log_probs[i] += log_probs_step[i, raw_token]

                if not active.any():
                    break

        # 计算当前样本在当前温度下的 PPL
        ppl_lis = []
        for i in range(bsz):
            if lengths[i] > 0:
                ppl = torch.exp(-cum_log_probs[i] / lengths[i]).item()
            else:
                ppl = float('inf')
            ppl_lis.append(ppl)

        # 后处理并保存当前样本结果
        df_now = pd.DataFrame({
            'rna_sequence': [id_seq_to_string(seqs_tokens[i]).rstrip('N') for i in range(bsz)],
            'ppl': ppl_lis,
            'protseq': [prot_seq for _ in range(bsz)],
            'temperature': [temp for _ in range(bsz)] # 记录温度列
        })
        df_all.append(df_now)

    # 每个温度跑完后，保存对应的 CSV
    final_df = pd.concat(df_all, ignore_index=True)
    import os
    os.makedirs("/data/ymxue/p4_protna/code/A_review/t3_gen/res2_gen_rnas/", exist_ok=True)
    output_path = f"/data/ymxue/p4_protna/code/A_review/t3_gen/res2_gen_rnas/gen_rna_temp{temp}.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved results for temp {temp} to {output_path}")
