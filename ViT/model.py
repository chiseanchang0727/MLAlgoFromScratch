import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Split the image into patches and then embed them.
    Parameters: 
        - image_size: int, the size of the image (it is a square).
        - patch_size: int, the size of the patch (it is a square).
        - in_channels: int, the number of input channels. grayscale = 1, RAG = 3.
        - embed_dim: int, the dimension of the embedding.
        
    Attritubtes:
        - n_patches: int, the number of patches inside the image.
        - proj: nn.Conv2d, the convolutional layer that does both the splitting and the embedding.
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
            - x: torch.Tensor, (n_samples, in_channels, img_size(height/width), img_size(height/width))
        Returns:
        torch.Tensor, (n_samples, n_patches, embed_dim)
        """
        
        x = self.proj(x) # (n_samples, embed_dim, image_size // patch_size, image_size // patch_size): reduce the spatial size by patch_size and increase the depth by embed_dim
        x = x.flatten(2) # (n_samples, embed_dim, n_patches): merge the dimesions 2 and 3 into a single dimension. Note that (image_size // patch_size)**2 = n_patches
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)
        
        return x