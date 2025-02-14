from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # save the particular tokens
        self.sos_token  = torch.Tensor((tokenizer_src.token_to_id(['SOS'])), dtype=torch.int64)  # usually the vocab is longer than 32-bit, so we use int64
        self.eos_token  = torch.Tensor((tokenizer_src.token_to_id(['EOS'])), dtype=torch.int64) 
        self.pad_token  = torch.Tensor((tokenizer_src.token_to_id(['PAD'])), dtype=torch.int64) 
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        encoded_input_tokens = self.tokenizer_src.encode(src_text).ids
        decoded_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # Pad the sentence to fill for reaching the seq_len, because the model input is fixed
        encoded_num_padding_tokens = self.seq_len - len(encoded_input_tokens) - 2 # 2 is <SOS> and <EOS>, those two will be added and don't need to be padded
        decoded_num_padding_tokens = self.seq_len - len(decoded_input_tokens) - 1 # Only <SOS> is added in the decoder side, EOS will be predicted
        
        if encoded_num_padding_tokens < 0 or decoded_num_padding_tokens <0:
            raise ValueError('Sentence is too long.')
        
        """
        Construct the encoded input sequence:
        Example:
        If encoded_input_tokens = [101, 2009, 2003, 1037, 2204, 2154, 102], 
        sos_token = torch.tensor([0]), eos_token = torch.tensor([1]), pad_token = 2, encoded_num_padding_tokens = 3,
        Then the final encoded_input will be:
        [<SOS>, 101, 2009, 2003, 1037, 2204, 2154, 102, <EOS>, <PAD>, <PAD>, <PAD>]
        """
        encoded_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoded_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoded_num_padding_tokens, dtype=torch.int64)
            ]
        )