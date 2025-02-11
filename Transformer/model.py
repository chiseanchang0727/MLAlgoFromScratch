import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        """
        d_model: the dimension of embedding vector
        vocab_size: how many words in the vocabulary
        """
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        """
        The reason we increase the embedding values before the addition is to make the positional encoding relatively smaller. 
        This means the original meaning in the embedding vector won’t be lost when we add them together.
        """
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Use torch.exp(math.log(10000.0)) to get a more stable result but the value is the same as multiplying 10000.0
        """
        # Using torch.exp
        exp_values = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))

        # Using torch.pow
        pow_values = torch.pow(torch.tensor(10000.0), torch.arange(0, d_model, 2).float() / d_model)

        # Check if both are the same
        print(torch.allclose(exp_values, pow_values))  # Should be True

        Since torch.exp() is more efficient than torch.pow()
        Because 
            torch.exp(x) directly computes the exponential in-place without requiring additional buffers.
            torch.pow(a, b) is actually decompose to torch.exp(b * math.log(a)) 
            torch.pow(a, b) may allocate additional temporary storage for ln(a), the multiplication, and then the exponentiation.
            torch.pow() has to deal with negative bases, fractional exponents, special case

        """
        div_term = torch.exp(torch.arange(0, d_model).float() * (math.log(10000.0) / d_model))

        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        """
        pe.unsqueeze(1) -> (seq_len, 1, d_model)
        pe.unsqueeze(-1) -> (seq_len, d_model, 1)
        """
        pe = pe.unsqueeze(0) # (1, seq_len, d_model): add a batch dimensoin for batched sentence input

        """
        In PyTorch modules, parameters are stored in the form of an OrderedDict, and they are classified into two types: nn.Parameter and buffers. 
        The difference is that during optimizer.step(), nn.Parameter gets updated, whereas buffers do not.
        """
        self.register_buffer('pe', pe)

    def forward(self, x):
        # slice the seq_len out, keep others the same
        x = x + self.pe[:, :x.shape[1], :].required_grad(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        
        # eps: Epsilon is for increasing the numerical stability(don't want too big/small) and avoid 0 in dominator
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.beta = nn.Parameter(torch.zeros(1)) # Added
        
    def forward(self, x):
        # dim=-1: means aggregate the value along with embeddings, i.e. d_model
        # use keepdim=True to maintain the original shape to avoid dimension mismatch
        mean = x.mean(dim=-1, keepdim=True) 
        std = x.std(dim=-1, keepdim=True)
        
        return self.alpha * mean / (std + self.eps) + self.beta
    
    
    
class MultiHeadAttetionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: int) -> None:
        """
        d_model: the size of embedding vector
        h: the number of head
        """
        super().__init__()
        self.d_model = d_model
        
        """
        Note that the QKV matrix is splitted along with the embedding dimension not the seq dimension,
        which means each head have access to full sentence but different part of the sequence.
        """
        self.h = h
        assert d_model % h == 0, "d_model is not dividable by h"
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # For applying the multiplication of query and the last two element in key, transpose the last two elements in key
        attention_score = (query @ key.transpose(-2 ,-1)) / math.sqrt(d_k)

        # For hiding some interaction between words
        if mask is not None:
            attention_score.masked_fill(mask==0, -1e10)
        attention_score = attention_score.softmask(dim = -1) # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        """
        mask: controlling information flow during training and inference, preventing attention to padding tokens and future imformation leakage
        """
        query = self.w_q(q) # (Batch, seq_len, d_moel) -> (Batch, seq_len, d_moel) 
        key = self.w_k(k) # (Batch, seq_len, d_moel) -> (Batch, seq_len, d_moel) 
        value = self.w_v(v) # (Batch, seq_len, d_moel) -> (Batch, seq_len, d_moel) 

        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_model) -> (Batch, h, seq_len, d_model)
        # Which means, each head will watch the full sentence but a smaller part of embedding
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores =MultiHeadAttetionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2)

        return 