import torch
class CharTokenizer:
    def __init__(self, text_data=None, special_tokens=None):
        """
        text_data: 包含所有可能出现的字符的字符串或列表
        special_tokens: 特殊符号列表，如 ['<pad>', '<sos>', '<eos>']
        """
        if special_tokens is None:
            special_tokens = ['<pad>', '<sos>', '<eos>']
            
        self.special_tokens = special_tokens
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        
        # 1. 建立词表
        # 收集所有唯一字符
        unique_chars = sorted(list(set(text_data))) if text_data else []
        # 组合: 特殊符号 + 普通字符
        self.vocab = self.special_tokens + unique_chars
        
        # 2. 建立映射关系
        self.char2id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id2char = {i: ch for i, ch in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        self.pad_id = self.char2id[self.pad_token]
        self.sos_id = self.char2id[self.sos_token]
        self.eos_id = self.char2id[self.eos_token]

    def encode(self, text, add_special_tokens=True):
        """把字符串转成 ID 列表"""
        ids = [self.char2id.get(c, self.pad_id) for c in text] # 找不到就返回PAD
        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """把 ID 列表转回字符串"""
        res = []
        for i in ids:
            if isinstance(i, torch.Tensor): i = i.item() # 处理 Tensor
            char = self.id2char.get(i, '')
            
            if skip_special_tokens:
                if char in self.special_tokens:
                    continue
            res.append(char)
        return "".join(res)