import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from transformers import PreTrainedTokenizer
import json

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file="/data/ymxue/p4_protna/code/vocab.json", **kwargs):
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__(**kwargs)

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def _add_tokens(self, new_tokens, special_tokens=False):
        num_added_toks = 0
        for token in new_tokens:
            if token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[token] = new_id
                self.ids_to_tokens[new_id] = token
                num_added_toks += 1
        return num_added_toks

    def encode(self, text, max_length=None, padding=False, truncation=False):
        tokens = self._tokenize(text)
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        if padding and max_length is not None:
            tokens += [self.pad_token] * (max_length - len(tokens))
        return [self._convert_token_to_id(token) for token in tokens]

from multimolecule import RnaTokenizer, RnaFmModel
from esm.tokenization import EsmSequenceTokenizer
from esm.sdk.api import ESMProtein, LogitsConfig
class ProteinRNADataset(Dataset):
    def __init__(self, csv_file, max_prot_len=1024, max_rna_len=512):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param tokenizer_prot: 蛋白质序列的分词器实例
        :param tokenizer_rna: RNA 序列的分词器实例
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        :param mode: 数据集模式，"pretrain" 或 "inference"
        :param data_type: 数据类型，"protein", "rna", 或 "both"
        """
        self.data = pd.read_csv(csv_file)
        # self.tokenizer = tokenizer
        self.rna_tokenizer = RnaTokenizer.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        self.prot_tokenizer = EsmSequenceTokenizer()

        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列
        prot_seq, rna_seq = self.data.iloc[idx][["prot_seq", "rna_seq"]]

        # 根据数据类型选择处理方式
        # 同时处理蛋白质和 RNA 序列
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, self.prot_tokenizer, mode="prot")
        rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, self.rna_tokenizer, mode="rna")
  
        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.prot_tokenizer.pad_token_id)
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = self._generate_padding_mask(rna_input_ids, self.rna_tokenizer.pad_token_id)
        rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)

        # 对序列进行掩码处理
        prot_masked_ids, prot_labels = self._mask_sequence(prot_input_ids, self.prot_tokenizer)
        rna_masked_ids, rna_labels = self._mask_sequence(rna_input_ids, self.rna_tokenizer)

        return {
            "prot_input_ids": torch.tensor(prot_masked_ids, dtype=torch.long),
            "prot_labels": torch.tensor(prot_labels, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_masked_ids, dtype=torch.long),
            "rna_labels": torch.tensor(rna_labels, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask
        }
        

    def _process_sequence(self, sequence, max_length, tokenizer, mode="prot"):
        if mode == "prot":
            prot = ESMProtein(sequence=sequence)
            sequence = prot.sequence
        tokens = tokenizer.encode(
            sequence,
            max_length=max_length,
            padding="max_length",       
            truncation=True     
        )
        if mode == "prot":
            # cls_token_id = tokenizer.cls_token_id
            # eos_token_id = tokenizer.eos_token_id
            # try:
            #     cls_index = tokens.index(cls_token_id)
            #     eos_index = tokens.index(eos_token_id)
            
            # except ValueError:
            #     raise ValueError("Tokenizer output does not contain [CLS] or [EOS] tokens.")
            # tokens = tokens[cls_index + 1:eos_index]  
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        return tokens
    
    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [1 if token_id != pad_token_id else 0 for token_id in input_ids]

    def _mask_sequence(self, input_ids, tokenizer):
        """
        BERT 风格的掩码策略：随机掩盖 15% 的 token。
        :param input_ids: 输入序列的 token IDs
        :param tokenizer: 分词器实例
        :return: 掩盖后的输入 ID 和对应的标签
        """
        masked_ids = input_ids.copy()
        labels = [-100] * len(input_ids)  # -100 表示忽略该位置的损失计算
        pad_token_id = tokenizer.pad_token_id
        mask_token_id = tokenizer.mask_token_id

        for i, token_id in enumerate(input_ids):
            if token_id == pad_token_id:  # 跳过填充标记
                continue
            if random.random() < 0.15:  # 15% 的概率掩盖
                labels[i] = token_id  # 标签为原始 token
                if random.random() < 0.8:  # 80% 的时间替换为 [MASK]
                    masked_ids[i] = mask_token_id
                elif random.random() < 0.5:  # 10% 的时间替换为随机 token
                    masked_ids[i] = random.choice(list(tokenizer.vocab.values()))
                # 剩余 10% 的时间保持不变
        return masked_ids, labels

class finetune_decoder(Dataset):
    def __init__(self, csv_file, max_prot_len=1024, max_rna_len=512):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param tokenizer_prot: 蛋白质序列的分词器实例
        :param tokenizer_rna: RNA 序列的分词器实例
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        :param mode: 数据集模式，"pretrain" 或 "inference"
        :param data_type: 数据类型，"protein", "rna", 或 "both"
        """
        self.data = pd.read_csv(csv_file)
        # self.tokenizer = tokenizer
        self.rna_tokenizer = RnaTokenizer.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        self.prot_tokenizer = EsmSequenceTokenizer()

        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列
        prot_seq, rna_seq = self.data.iloc[idx][["prot", "rna"]]

        # 根据数据类型选择处理方式
        # 同时处理蛋白质和 RNA 序列
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, self.prot_tokenizer, mode="prot")
        rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, self.rna_tokenizer, mode="rna")
  
        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.prot_tokenizer.pad_token_id)
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = self._generate_padding_mask(rna_input_ids, self.rna_tokenizer.pad_token_id)
        rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)

        # 对序列进行掩码处理
        prot_labels = self._mask_sequence(prot_input_ids, self.prot_tokenizer)
        rna_labels = self._mask_sequence(rna_input_ids, self.rna_tokenizer)

        return {
            "prot_input_ids": torch.tensor(prot_input_ids, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_input_ids, dtype=torch.long),
            "prot_labels": torch.tensor(prot_labels, dtype=torch.long),
            "rna_labels": torch.tensor(rna_labels, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask
        }
        

    def _process_sequence(self, sequence, max_length, tokenizer, mode="prot"):
        if mode == "prot":
            prot = ESMProtein(sequence=sequence)
            sequence = prot.sequence
        tokens = tokenizer.encode(
            sequence,
            max_length=max_length,
            padding="max_length",       
            truncation=True     
        )
        if mode == "prot":
            # cls_token_id = tokenizer.cls_token_id
            # eos_token_id = tokenizer.eos_token_id
            # try:
            #     cls_index = tokens.index(cls_token_id)
            #     eos_index = tokens.index(eos_token_id)
            
            # except ValueError:
            #     raise ValueError("Tokenizer output does not contain [CLS] or [EOS] tokens.")
            # tokens = tokens[cls_index + 1:eos_index]  
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        return tokens
    
    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [1 if token_id != pad_token_id else 0 for token_id in input_ids]

    def _mask_sequence(self, input_ids, tokenizer):
        # masked_ids = input_ids.copy()
        labels = [-100] * len(input_ids)  # -100 表示忽略该位置的损失计算
        pad_token_id = tokenizer.pad_token_id

        for i, token_id in enumerate(input_ids):
            if token_id == pad_token_id:  # 跳过填充标记
                continue
            else:
                labels[i] = token_id
        return labels

class DownstreamDataset_onehot(Dataset):
    def __init__(self, csv_file, max_prot_len=1024, max_rna_len=512):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        """
        self.data = pd.read_csv(csv_file)

        # 定义最大长度
        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len

        # 定义统一的词汇表，区分 RNA 和蛋白质
        self.vocab = {
            # 蛋白质词汇表
            "PROT_A": 1, "PROT_C": 2, "PROT_D": 3, "PROT_E": 4, "PROT_F": 5,
            "PROT_G": 6, "PROT_H": 7, "PROT_I": 8, "PROT_K": 9, "PROT_L": 10,
            "PROT_M": 11, "PROT_N": 12, "PROT_P": 13, "PROT_Q": 14, "PROT_R": 15,
            "PROT_S": 16, "PROT_T": 17, "PROT_V": 18, "PROT_W": 19, "PROT_Y": 20,
            # RNA 词汇表
            "RNA_A": 21, "RNA_C": 22, "RNA_G": 23, "RNA_U": 24,
            # 特殊符号
            "<PAD>": 0,  # 填充符号
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列以及目标值
        prot_seq, rna_seq, value = self.data.iloc[idx][["prot_seq", "rna_seq", "deltaG"]]

        # 处理蛋白质和 RNA 序列
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, prefix="PROT_")
        rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, prefix="RNA_")

        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.vocab["<PAD>"])
        rna_padding_mask = self._generate_padding_mask(rna_input_ids, self.vocab["<PAD>"])

        # 将 padding mask 转换为布尔张量
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)

        return {
            "prot_input_ids": torch.tensor(prot_input_ids, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_input_ids, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask,
            "value": torch.tensor(value, dtype=torch.float),
        }

    def _process_sequence(self, sequence, max_length, prefix="PROT_"):
        """
        处理单个序列：将其转换为 token IDs，并进行填充或截断。
        :param sequence: 输入的蛋白质或 RNA 序列
        :param max_length: 最大长度
        :param prefix: 区分 RNA 和蛋白质的前缀（如 "PROT_" 或 "RNA_"）
        :return: 填充或截断后的 token IDs 列表
        """
        # 将序列中的每个字符映射到词汇表 ID，带上前缀
        tokens = [self.vocab.get(f"{prefix}{char}", self.vocab["<PAD>"]) for char in sequence]

        # 截断或填充到指定长度
        if len(tokens) > max_length:
            tokens = tokens[:max_length]  # 截断
        else:
            tokens += [self.vocab["<PAD>"]] * (max_length - len(tokens))  # 填充

        return tokens

    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [token_id != pad_token_id for token_id in input_ids]

class DownstreamDataset_onehot_token(Dataset):
    def __init__(self, csv_file, max_prot_len=1024, max_rna_len=512):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        """
        self.data = pd.read_csv(csv_file)

        # 定义最大长度
        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len

        # 定义统一的词汇表，区分 RNA 和蛋白质
        self.vocab = {
            # 蛋白质词汇表
            "PROT_A": 1, "PROT_C": 2, "PROT_D": 3, "PROT_E": 4, "PROT_F": 5,
            "PROT_G": 6, "PROT_H": 7, "PROT_I": 8, "PROT_K": 9, "PROT_L": 10,
            "PROT_M": 11, "PROT_N": 12, "PROT_P": 13, "PROT_Q": 14, "PROT_R": 15,
            "PROT_S": 16, "PROT_T": 17, "PROT_V": 18, "PROT_W": 19, "PROT_Y": 20,
            # RNA 词汇表
            "RNA_A": 21, "RNA_C": 22, "RNA_G": 23, "RNA_U": 24,
            # 特殊符号
            "<PAD>": 0,  # 填充符号
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列以及目标值
        rna_seq, rna_bind, prot_seq, prot_bind = self.data.iloc[idx][["rna", "rna_bind", "prot", "prot_bind"]]

        # 处理蛋白质和 RNA 序列
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, prefix="PROT_")
        rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, prefix="RNA_")

        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.vocab["<PAD>"])
        rna_padding_mask = self._generate_padding_mask(rna_input_ids, self.vocab["<PAD>"])

        # 将 padding mask 转换为布尔张量
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)

        prot_bind = [int(char) for char in prot_bind]
        rna_bind = [int(char) for char in rna_bind]
        prot_bind = prot_bind + [6] * (self.max_prot_len - len(prot_bind))
        rna_bind = rna_bind + [6] * (self.max_rna_len - len(rna_bind))

        return {
            "prot_input_ids": torch.tensor(prot_input_ids, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_input_ids, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask,
            "prot_bind": torch.tensor(prot_bind, dtype=torch.long),
            "rna_bind": torch.tensor(rna_bind, dtype=torch.long)
        }

    def _process_sequence(self, sequence, max_length, prefix="PROT_"):
        """
        处理单个序列：将其转换为 token IDs，并进行填充或截断。
        :param sequence: 输入的蛋白质或 RNA 序列
        :param max_length: 最大长度
        :param prefix: 区分 RNA 和蛋白质的前缀（如 "PROT_" 或 "RNA_"）
        :return: 填充或截断后的 token IDs 列表
        """
        # 将序列中的每个字符映射到词汇表 ID，带上前缀
        tokens = [self.vocab.get(f"{prefix}{char}", self.vocab["<PAD>"]) for char in sequence]

        # 截断或填充到指定长度
        if len(tokens) > max_length:
            tokens = tokens[:max_length]  # 截断
        else:
            tokens += [self.vocab["<PAD>"]] * (max_length - len(tokens))  # 填充

        return tokens

    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [token_id != pad_token_id for token_id in input_ids]


class DownstreamDataset(Dataset):
    def __init__(self, csv_file, max_prot_len=1024, max_rna_len=512, mode="pretrain", data_type="both"):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param tokenizer_prot: 蛋白质序列的分词器实例
        :param tokenizer_rna: RNA 序列的分词器实例
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        :param mode: 数据集模式，"pretrain" 或 "inference"
        :param data_type: 数据类型，"protein", "rna", 或 "both"
        """
        self.data = pd.read_csv(csv_file)
        # self.tokenizer = tokenizer
        self.rna_tokenizer = RnaTokenizer.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        self.prot_tokenizer = EsmSequenceTokenizer()

        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len
        self.mode = mode  # 模式：预训练或推理
        self.data_type = data_type  # 数据类型："protein", "rna", 或 "both"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列
        prot_seq, rna_seq, value = self.data.iloc[idx][["prot_seq", "rna_seq", "deltaG"]]

        # 根据数据类型选择处理方式
        # 同时处理蛋白质和 RNA 序列
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, self.prot_tokenizer, mode="prot")
        rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, self.rna_tokenizer, mode="rna")
  
        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.prot_tokenizer.pad_token_id)
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = self._generate_padding_mask(rna_input_ids, self.rna_tokenizer.pad_token_id)
        rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)
        # padding_mask = prot_padding_mask.unsqueeze(1) & rna_padding_mask.unsqueeze(0)


        return {
            "prot_input_ids": torch.tensor(prot_input_ids, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_input_ids, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask,
            "value": torch.tensor(value, dtype=torch.float),
        }
        

    def _process_sequence(self, sequence, max_length, tokenizer, mode="prot"):
        if mode == "prot":
            prot = ESMProtein(sequence=sequence)
            sequence = prot.sequence
        tokens = tokenizer.encode(
            sequence,
            max_length=max_length,
            padding="max_length",       
            truncation=True     
        )
        if mode == "prot":
            # cls_token_id = tokenizer.cls_token_id
            # eos_token_id = tokenizer.eos_token_id
            # try:
            #     cls_index = tokens.index(cls_token_id)
            #     eos_index = tokens.index(eos_token_id)
            
            # except ValueError:
            #     raise ValueError("Tokenizer output does not contain [CLS] or [EOS] tokens.")
            # tokens = tokens[cls_index + 1:eos_index]  
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        return tokens
    
    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [1 if token_id != pad_token_id else 0 for token_id in input_ids]

class DownstreamDS_token(Dataset):
    def __init__(self, csv_file, max_prot_len=1024, max_rna_len=512):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param tokenizer_prot: 蛋白质序列的分词器实例
        :param tokenizer_rna: RNA 序列的分词器实例
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        :param mode: 数据集模式，"pretrain" 或 "inference"
        :param data_type: 数据类型，"protein", "rna", 或 "both"
        """
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.rna_tokenizer = RnaTokenizer.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        self.prot_tokenizer = EsmSequenceTokenizer()

        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列
        rna_seq, rna_bind, prot_seq, prot_bind = self.data.iloc[idx][["rna", "rna_bind", "prot", "prot_bind"]]

        # 根据数据类型选择处理方式
        ### 注意esm的编码方式：[CLS] + protein + [SEP]， 需要考虑前后的[CLS]和[SEP]
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, self.prot_tokenizer, mode="prot")
        rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, self.rna_tokenizer, mode="rna")
        prot_bind = [int(char) for char in prot_bind]
        rna_bind = [int(char) for char in rna_bind]
        # # binding_site_labels前后各加一个6
        prot_bind = [6] + prot_bind + [6]
        rna_bind = [6] + rna_bind + [6]
        prot_bind.extend([6] * (self.max_prot_len - len(prot_bind)))
        rna_bind.extend([6] * (self.max_rna_len - len(rna_bind)))

        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.prot_tokenizer.pad_token_id)
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = self._generate_padding_mask(rna_input_ids, self.rna_tokenizer.pad_token_id)
        rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)

        return {
            "prot_input_ids": torch.tensor(prot_input_ids, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_input_ids, dtype=torch.long),
            "prot_bind": torch.tensor(prot_bind, dtype=torch.long),
            "rna_bind": torch.tensor(rna_bind, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask,
        }
        

    def _process_sequence(self, sequence, max_length, tokenizer, mode):
        if mode == "prot":
            prot = ESMProtein(sequence=sequence)
            sequence = prot.sequence
        tokens = tokenizer.encode(
            sequence,
            max_length=max_length,
            padding="max_length",       
            truncation=True     
        )
        if mode == "prot":  
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        return tokens
    
    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [1 if token_id != pad_token_id else 0 for token_id in input_ids]

from scipy.sparse import csr_matrix
class DownstreamDS_2d(Dataset):
    def __init__(self, csv_file, max_prot_len=1024, max_rna_len=512):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param tokenizer_prot: 蛋白质序列的分词器实例
        :param tokenizer_rna: RNA 序列的分词器实例
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        :param mode: 数据集模式，"pretrain" 或 "inference"
        :param data_type: 数据类型，"protein", "rna", 或 "both"
        """
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.rna_tokenizer = RnaTokenizer.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        self.prot_tokenizer = EsmSequenceTokenizer()

        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列
        rna_seq, prot_seq, bind = self.data.iloc[idx][["rna", "prot", "bind"]]

        # 根据数据类型选择处理方式
        ### 注意esm的编码方式：[CLS] + protein + [SEP]， 需要考虑前后的[CLS]和[SEP]
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, self.prot_tokenizer, mode="prot")
        rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, self.rna_tokenizer, mode="rna")

        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.prot_tokenizer.pad_token_id)
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = self._generate_padding_mask(rna_input_ids, self.rna_tokenizer.pad_token_id)
        rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)

        # generate 2d padding mask
        # 0 + seq*1 + 0*(max-seq)
        tod_prot = torch.tensor([0] + [1]*len(prot_seq) + [0]*(self.max_prot_len-len(prot_seq)-1), dtype=torch.bool)
        tod_rna = torch.tensor([0] + [1]*len(rna_seq) + [0]*(self.max_rna_len-len(rna_seq)-1), dtype=torch.bool)
        tod_valid_mask = tod_prot.unsqueeze(1) & tod_rna.unsqueeze(0)

        entries = [list(map(int, entry.split(','))) for entry in bind.split(';')]
        rows, cols, values = zip(*entries)
        shape = (1024, 512)  # 矩阵的形状需要提前知道
        sparse_matrix = csr_matrix((values, (rows, cols)), shape=shape)
        dense_bind = torch.tensor(sparse_matrix.toarray())

        return {
            "prot_input_ids": torch.tensor(prot_input_ids, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_input_ids, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask,
            "bind": dense_bind,
            "bind_valid": tod_valid_mask
        }
        

    def _process_sequence(self, sequence, max_length, tokenizer, mode):
        if mode == "prot":
            prot = ESMProtein(sequence=sequence)
            sequence = prot.sequence
        tokens = tokenizer.encode(
            sequence,
            max_length=max_length,
            padding="max_length",       
            truncation=True     
        )
        if mode == "prot":
            # cls_token_id = tokenizer.cls_token_id
            # eos_token_id = tokenizer.eos_token_id
            # try:
            #     cls_index = tokens.index(cls_token_id)
            #     eos_index = tokens.index(eos_token_id)
            
            # except ValueError:
            #     raise ValueError("Tokenizer output does not contain [CLS] or [EOS] tokens.")
            # tokens = tokens[cls_index + 1:eos_index]  
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        return tokens
    
    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [1 if token_id != pad_token_id else 0 for token_id in input_ids]
    
import random
class DownstreamDS_token_single(Dataset):
    def __init__(self, csv_file, max_prot_len=1024):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param tokenizer_prot: 蛋白质序列的分词器实例
        :param tokenizer_rna: RNA 序列的分词器实例
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        :param mode: 数据集模式，"pretrain" 或 "inference"
        :param data_type: 数据类型，"protein", "rna", 或 "both"
        """
        super().__init__()
        self.data = pd.read_csv(csv_file)
        # self.rna_tokenizer = RnaTokenizer.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        self.prot_tokenizer = EsmSequenceTokenizer()

        self.max_prot_len = max_prot_len
        # self.max_rna_len = max_rna_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列
        prot_seq, prot_bind = self.data.iloc[idx][["sequence", "label"]]

        # 根据数据类型选择处理方式
        ### 注意esm的编码方式：[CLS] + protein + [SEP]， 需要考虑前后的[CLS]和[SEP]
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, self.prot_tokenizer, mode="prot")
        # rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, self.rna_tokenizer, mode="rna")
        prot_bind = [int(char) for char in prot_bind]
        # rna_bind = [int(char) for char in rna_bind]
        # # binding_site_labels前后各加一个6
        prot_bind = [6] + prot_bind + [6]
        # rna_bind = [6] + rna_bind + [6]
        prot_bind.extend([6] * (self.max_prot_len - len(prot_bind)))
        # rna_bind.extend([6] * (self.max_rna_len - len(rna_bind)))

        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.prot_tokenizer.pad_token_id)
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = self._generate_rna_padding_mask(512)
        # rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)

        return {
            "prot_input_ids": torch.tensor(prot_input_ids, dtype=torch.long),
            # "rna_input_ids": torch.tensor(rna_input_ids, dtype=torch.long),
            "prot_bind": torch.tensor(prot_bind, dtype=torch.long),
            # "rna_bind": torch.tensor(rna_bind, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask,
        }
        
    def _generate_rna_padding_mask(self, max_length):
        seq_length = random.randint(5, 250)
        padding_mask = [True] * seq_length + [False] * (max_length - seq_length)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
        return padding_mask

    def _process_sequence(self, sequence, max_length, tokenizer, mode):
        if mode == "prot":
            prot = ESMProtein(sequence=sequence)
            sequence = prot.sequence
        tokens = tokenizer.encode(
            sequence,
            max_length=max_length,
            padding="max_length",       
            truncation=True     
        )
        if mode == "prot":
            # cls_token_id = tokenizer.cls_token_id
            # eos_token_id = tokenizer.eos_token_id
            # try:
            #     cls_index = tokens.index(cls_token_id)
            #     eos_index = tokens.index(eos_token_id)
            
            # except ValueError:
            #     raise ValueError("Tokenizer output does not contain [CLS] or [EOS] tokens.")
            # tokens = tokens[cls_index + 1:eos_index]  
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        return tokens
    
    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [1 if token_id != pad_token_id else 0 for token_id in input_ids]


class DownstreamDataset_apta(Dataset):
    def __init__(self, csv_file, max_prot_len=1024, max_rna_len=512, mode="pretrain", data_type="both"):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param tokenizer_prot: 蛋白质序列的分词器实例
        :param tokenizer_rna: RNA 序列的分词器实例
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        :param mode: 数据集模式，"pretrain" 或 "inference"
        :param data_type: 数据类型，"protein", "rna", 或 "both"
        """
        self.data = pd.read_csv(csv_file)
        # self.tokenizer = tokenizer
        self.rna_tokenizer = RnaTokenizer.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        self.prot_tokenizer = EsmSequenceTokenizer()

        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len
        self.mode = mode  # 模式：预训练或推理
        self.data_type = data_type  # 数据类型："protein", "rna", 或 "both"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列
        prot_seq = "EFRDCAEVFKSGHTTNGIYTLTFPNSTEEIKAYCDMEAGGGGWTIIQRREDGSVDFQRTWKEYKVGFGNPS" \
                    "GEYWLGNEFVSQLTNQQRYVLKIHLKDWEGNEAYSLYEHFYLSSEELNYRIHLKGLTGTAGKISSISQPG" \
                    "NDFSTKDGDNDKCICKCSQMLTGGWWFDACGPSNLNGMYYPQRQNTNKFNGIKWYYWKGSGYSLKATTMMIRPAD"
        rna_seq, value = self.data.iloc[idx][["seq", "zrank_norm"]]

        # 根据数据类型选择处理方式
        # 同时处理蛋白质和 RNA 序列
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, self.prot_tokenizer, mode="prot")
        rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, self.rna_tokenizer, mode="rna")
  
        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.prot_tokenizer.pad_token_id)
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = self._generate_padding_mask(rna_input_ids, self.rna_tokenizer.pad_token_id)
        rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)
        # padding_mask = prot_padding_mask.unsqueeze(1) & rna_padding_mask.unsqueeze(0)


        return {
            "prot_input_ids": torch.tensor(prot_input_ids, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_input_ids, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask,
            "value": torch.tensor(value, dtype=torch.float),
        }
        

    def _process_sequence(self, sequence, max_length, tokenizer, mode="prot"):
        if mode == "prot":
            prot = ESMProtein(sequence=sequence)
            sequence = prot.sequence
        tokens = tokenizer.encode(
            sequence,
            max_length=max_length,
            padding="max_length",       
            truncation=True     
        )
        if mode == "prot":
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        return tokens
    
    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [1 if token_id != pad_token_id else 0 for token_id in input_ids]

class DownstreamDataset_apta_class(Dataset):
    def __init__(self, csv_file, tokenizer, max_prot_len=1024, max_rna_len=512, mode="pretrain", data_type="both"):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param tokenizer_prot: 蛋白质序列的分词器实例
        :param tokenizer_rna: RNA 序列的分词器实例
        :param max_prot_len: 蛋白质序列的最大长度（填充后）
        :param max_rna_len: RNA 序列的最大长度（填充后）
        :param mode: 数据集模式，"pretrain" 或 "inference"
        :param data_type: 数据类型，"protein", "rna", 或 "both"
        """
        self.data = pd.read_csv(csv_file)
        # self.tokenizer = tokenizer
        self.rna_tokenizer = RnaTokenizer.from_pretrained("/data/ymxue/p4_protna/code/fm_model/rnafm")
        self.prot_tokenizer = EsmSequenceTokenizer()

        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len
        self.mode = mode  # 模式：预训练或推理
        self.data_type = data_type  # 数据类型："protein", "rna", 或 "both"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取蛋白质和 RNA 序列
        prot_seq, rna_seq, value = self.data.iloc[idx][["prot", "rna", "label"]]

        # 根据数据类型选择处理方式
        # 同时处理蛋白质和 RNA 序列
        prot_input_ids = self._process_sequence(prot_seq, self.max_prot_len, self.prot_tokenizer, mode="prot")
        rna_input_ids = self._process_sequence(rna_seq, self.max_rna_len, self.rna_tokenizer, mode="rna")
  
        # 生成 padding mask
        prot_padding_mask = self._generate_padding_mask(prot_input_ids, self.prot_tokenizer.pad_token_id)
        prot_padding_mask = torch.tensor(prot_padding_mask, dtype=torch.bool)
        rna_padding_mask = self._generate_padding_mask(rna_input_ids, self.rna_tokenizer.pad_token_id)
        rna_padding_mask = torch.tensor(rna_padding_mask, dtype=torch.bool)
        # padding_mask = prot_padding_mask.unsqueeze(1) & rna_padding_mask.unsqueeze(0)

        return {
            "prot_input_ids": torch.tensor(prot_input_ids, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_input_ids, dtype=torch.long),
            "prot_padding_mask": prot_padding_mask,
            "rna_padding_mask": rna_padding_mask,
            "value": torch.tensor(value, dtype=torch.float),
        }
        

    def _process_sequence(self, sequence, max_length, tokenizer, mode="prot"):
        if mode == "prot":
            prot = ESMProtein(sequence=sequence)
            sequence = prot.sequence
        tokens = tokenizer.encode(
            sequence,
            max_length=max_length,
            padding="max_length",       
            truncation=True     
        )
        # if mode == "prot":
        cls_token_id = tokenizer.cls_token_id
        eos_token_id = tokenizer.eos_token_id
        try:
            cls_index = tokens.index(cls_token_id)
            eos_index = tokens.index(eos_token_id)
        
        except ValueError:
            raise ValueError("Tokenizer output does not contain [CLS] or [EOS] tokens.")
        tokens = tokens[cls_index + 1:eos_index]  
        tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        return tokens
    
    def _generate_padding_mask(self, input_ids, pad_token_id):
        """
        根据输入的 token IDs 和 pad_token_id 生成 padding mask。
        :param input_ids: 输入序列的 token IDs
        :param pad_token_id: 分词器的 pad token ID
        :return: 一个布尔类型的 mask，形状与 input_ids 相同
        """
        return [1 if token_id != pad_token_id else 0 for token_id in input_ids]


def get_dataloader(csv_file, tokenizer, batch_size=32, shuffle=True):
    """
    构建 DataLoader
    :param csv_file: CSV 文件路径
    :param tokenizer: 自定义分词器实例
    :param batch_size: 批次大小
    :param shuffle: 是否打乱数据
    :return: DataLoader 实例
    """
    dataset = ProteinRNADataset(csv_file, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# 示例用法
if __name__ == '__main__':
    # 加载分词器
    tokenizer = CustomTokenizer()
    
    # 添加特殊标记
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "mask_token": "[MASK]"
    })
    
    # 数据集路径
    csv_file = "/data/ymxue/p4_protna/data/CoPRA/cleaned_data/pretrain.csv"
    
    # 构建 DataLoader
    dataloader = get_dataloader(csv_file, tokenizer, batch_size=32)
    
    # 打印一个批次的数据
    for i, batch in enumerate(dataloader):
        if i == 0:
            print("Protein Input IDs shape:", batch["prot_input_ids"].shape)
            print("Protein Attention Mask shape:", batch["prot_labels"].shape)
            print("RNA Input IDs shape:", batch["rna_input_ids"].shape)
            print("RNA Attention Mask shape:", batch["rna_labels"].shape)
            break