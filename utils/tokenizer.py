from transformers import PreTrainedTokenizer
import json

# 自定义一个简单的分词器类
class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file="/data/ymxue/p4_protna/code/vocab.json", **kwargs):
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__(**kwargs)

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _tokenize(self, text):
        # 将输入的文本字符串转换为一个字符列表
        # 参数:
        #   text (str): 需要被分词的文本字符串
        # 返回:
        #   list: 包含文本中所有字符的列表
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

if __name__ == '__main__':
    # 加载分词器
    tokenizer = CustomTokenizer(vocab_file="/data/ymxue/p4_protna/code/vocab.json")
    
    # 添加特殊标记
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "mask_token": "[MASK]"
    })
    
    # 测试分词器
    sequence = "ARNDCQEGHILKMFPSTWYVX*AU"
    
    # 编码并填充到最大长度 50
    tokens = tokenizer.encode(
        sequence,
        max_length=50,
        padding=True,
        truncation=True
    )
    print("Tokens:", tokens)
    print("Decoded:", tokenizer.decode(tokens, skip_special_tokens=False))