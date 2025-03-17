import torch
import torch.nn as nn

# Ref: https://www.youtube.com/watch?v=ovB0ddFtzzA&t=857s

# TODO: position encoding should be added


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds them.

    Parameters:
        image_size (int): size of the input image (assumed to be square).
        patch_size (int): size of each patch (assumed to be square).
        in_channels (int): number of input channels (1 for grayscale, 3 for RGB).
        embed_dim (int): dimension of the embedding.

    Attributes:
        n_patches (int): number of patches in the image.
        proj (nn.Conv2d): convolutional layer that performs both patch splitting and embedding.
    """

    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size # no overlap
        )
        
    def forward(self, x):
        """
        Run forward pass.
        
        Parameters:
            x: torch.Tensor, (n_samples, in_channels, img_size(height/width), img_size(height/width))
        Returns:
        torch.Tensor, (n_samples, n_patches, embed_dim)
        """
        
        x = self.proj(x) # (n_samples, embed_dim, image_size // patch_size, image_size // patch_size): reduce the spatial size by patch_size and increase the depth by embed_dim
        x = x.flatten(2) # (n_samples, embed_dim, n_patches): merge the dimesions 2 and 3 into a single dimension. Note that (image_size // patch_size)**2 = n_patches
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)
        
        return x
    
    
class Attention(nn.Module):
    """
    Attention Mechanism.

    Parameters:
        dim (int): input and output dimension per token feature.
        n_heads (int): number of attention heads.
        qkv_bias (bool): if True, includes bias in the query, key, and value projections.
        attn_p (float): dropout probability applied to the query, key, and value tensors.
        proj_p (float): dropout probability applied to the output tensor.

    Attributes:
        scale (float): normalization factor for the dot product.
        qkv (nn.Linear): linear transformation for the query, key, and value.
        proj (nn.Linear): maps the concatenated output of all attention heads into a new space.
        attn_drop (nn.Dropout): dropout layer applied to the attention scores.
        proj_drop (nn.Dropout): dropout layer applied to the final projection output.
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** 0.5 # Don't feed too large values into softmax, scale them down by the head_dim, actuall it's the sqrt(d_k) in "attention is all you need"
        
        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.atten_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim) # Wo in Transformer
        self.proj_drop = nn.Dropout(proj_p)
        
    @staticmethod
    def attention_score(q, k, v, scale, dropout):
        
        score = (q @ k.transpose(-2, -1)) / scale
        
        attn_score = torch.softmax(score, dim=-1)
        
        if dropout is not None:
            attn_score = dropout(attn_score)

        return (attn_score @ v), attn_score
        
        
    def forward(self, x):
        """
        Run forward pass.
        
        Parameters:
            x: torch.Tensor, (n_samples, n_pathces + 1, dim), +1 means there will be an extra learnable token for the class token.
        
        Returns:
            torch.Tensor, (n_samples, n_patches + 1, dim)
        """
        n_samples, n_tokens, dim = x.shape
        
        if dim != self.dim:
            raise ValueError(f'Input dim {dim} should be {self.dim}')
        
        query = self.w_q(x) # (n_samples, n_patches + 1, dim)
        key = self.w_k(x) # (n_samples, n_patches + 1, dim)
        value = self.w_v(x) # (n_samples, n_patches + 1, dim)
        
        # Reshape to (n_samples, n_heads, n_patches + 1, head_dim), n_tokens = n_pathes + 1 (1 stands for cls token)
        query = query.view(n_samples, n_tokens, self.heads, self.head_dim).transpose(1, 2) # n_tokens = n_patches + 1
        key = key.view(n_samples, n_tokens, self.heads, self.head_dim).transpose(1, 2) # n_tokens = n_patches + 1
        value = value.view(n_samples, n_tokens, self.heads, self.head_dim).transpose(1, 2) # n_tokens = n_patches + 1
        
        x, attn_score = Attention.attention_score(query, key, value, self.scale, self.atten_drop)

        # Reshape to (n_samples, n_patches + 1, dim)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.head_dim)

        """
        Note: 
            1. there is one more dropout compared with transformer
            2. proj is the projection layer for output, a.k.a Wo
        """
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module): # FeedForwardBlock
    """
    Multilayer Perceptron (MLP).

    Parameters:
        in_features (int): number of input features.
        hidden_features (int): number of nodes in the hidden layer.
        out_features (int): number of output features.
        dropout (float): dropout probability.

    Attributes:
        fc (nn.Linear): first linear layer.
        act (nn.GELU): GELU activation function.
        fc2 (nn.Linear): second linear layer.
        dropout (nn.Dropout): dropout layer.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Run forward pass.

        Parameters:
            x: torch.Tensor, (n_samples, n_patches+1, in_features)
        
        Returns:
            torch.Tensor, (n_samples, n_patches+1, out_features)
        """
        x = self.fc(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
    
"""
The purpose of Vit is classification, so it doesn't need decoder.(no need for generation, like text)
"""
class Block: # in ViT we only have encoder block
    """
    Transformer block.

    Parameters:
        dim (int): embedding dimension
        n_heads (int): number of attention heads
        mlp_ratio (float): determine the hidden dimension size of the MLP module with respect to `dim`
        qkv_bias (bool): if True then we include bias, otherwise not include
        dropout (float): dropout probability
    
    Attributes:
        norm1, norm2: layer normalization
        attn: attetion module
        mlp: MLP module
    """
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=True, attn_dropout: float=0., proj_drop: float=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.attn_block = Attention(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_dropout,
            proj_p=proj_drop
        )

        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )
    
    def forward(self, x):
        """
        Run forward pass.

        Parameters:
        x: torch.Tensor, (n_samples, n_patches+1, dim)
        
        Returns:
        torch.Tensor, (n_samples, n_patches+1, dim)
        """

        x = x + self.attn_block(self.norm1(x)) # Residual connection
        x = x + self.mlp(self.norm2(x))

        return x
    

class ViT(nn.Module):
    """
    Simplified implementation of the Vision Transformer.

    Parameters:
        img_size (int): height and width of the input image (assumed to be square).
        patch_size (int): height and width of each patch (assumed to be square).
        in_channels (int): number of input channels.
        n_class (int): number of classes.
        embed_dim (int): dimensionality of the token/patch embeddings.
        depth (int): number of blocks.
        n_heads (int): number of attention heads
        mlp_ratio (float): determines the hidden dimension of the 'MLP' module.
    """
