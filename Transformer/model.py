import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        """
        d_model: the dimension of embedding vector
        vocab_size: how many words in the vocabulary
        """
        super().__init__()
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))

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
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        
        # eps: Epsilon is for increasing the numerical stability(don't want too big/small) and avoid 0 in dominator
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # Multiplied
        self.beta = nn.Parameter(torch.zeros(features)) # Added
        
    def forward(self, x):
        # dim=-1: means aggregate the value along with embeddings, i.e. d_model
        # use keepdim=True to maintain the original shape to avoid dimension mismatch
        mean = x.mean(dim=-1, keepdim=True) 
        std = x.std(dim=-1, keepdim=True)
        
        return self.alpha * mean / (std + self.eps) + self.beta
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
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
        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2 ,-1)) / math.sqrt(d_k)

        # For hiding some interaction between words
        if mask is not None:
            attention_scores.masked_fill(mask==0, -1e10)
        
        # Apply the softmask to the last dimension
        attention_scores = torch.softmax(attention_scores, dim = -1)  # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        mask: controlling information flow during training and inference, preventing attention to padding tokens and future imformation leakage
        """
        query = self.w_q(q) # (Batch, seq_len, d_moel) -> (Batch, seq_len, d_moel) 
        key = self.w_k(k) # (Batch, seq_len, d_moel) -> (Batch, seq_len, d_moel) 
        value = self.w_v(v) # (Batch, seq_len, d_moel) -> (Batch, seq_len, d_moel) 

        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        # Which means, each head will watch the full sentence but a smaller part of embedding
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttetionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        # -1: PyTorch calculates this dimension automatically to ensure the total number of elements remains consistent. i.e. let pytorch figure out this dimension
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    """
    This is the code for "Add & Norm" layer
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)


    def forward(self, x, sublayer):
        """
        sublayer: previous layer, a callable object

        In the "attention is all you need", it should be self.norm(x + self.dropout(sublayer(x)))
        Pros: More stable training.
        Cons: Gradient flow may be challenging in deep layers (because normalization happens last, which can cause gradient shrinkage).

        but the paper (https://arxiv.org/pdf/2002.04745) suggest a way called Pre-Norm Residual Connection 
        Pros:
        Improves gradient flow, helping deeper transformers train more effectively.
        Reduces instability during training, especially when using deeper models.
        Cons:
        The model may be less robust to certain hyperparameter choices.
        Requires careful tuning of learning rates.

        Conclusion:
        Gradient Flow Issues in Deep Transformers:

            Post-Norm (original) can lead to vanishing gradients in deep transformers.
            Pre-Norm improves gradient propagation, making deep models more stable.
        Empirical Results:

        Studies (including Xiong et al., 2020) show that Pre-Norm transformers train faster and more stably than Post-Norm transformers, 
        especially for very deep networks (e.g., GPT-like architectures).
        """

        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttetionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block # It only focus on the input sentence itslef, so called self-attention
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # two residual connection in the encoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x : self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """
    For cross attention part, the key is from decoder while query and key are form encoder
    """

    def __init__(self, 
                 features: int,
                 self_attention_block: MultiHeadAttetionBlock, 
                 cross_attention_block: MultiHeadAttetionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features ,dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Cross attention has the same structure with multi-head attention block, the "cross" means the input comes from different sequence
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # (q, k, v, mask)
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """
    Project the embedding into vocabulary
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (Batch, seq_len, d_moel) ->  (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1) # Use log_softmax to numerical stability
    
class Transformer(nn.Module):

    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model=512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # Create embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create position encoding layer
    src_pos = PositionalEncoding(d_model, src_seq_len ,dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttetionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model ,encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttetionBlock(d_model, h, dropout)
        decoder_cross_attention_decoder_block = MultiHeadAttetionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_decoder_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer