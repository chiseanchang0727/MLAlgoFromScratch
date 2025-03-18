import numpy as np
import timm
import torch
from trial_mildlyoverfitted import ViT
import os

os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'


# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_tensor_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t1.detach().numpy()
    np.testing.assert_allclose(a1, a2)



model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name=model_name, pretrained=True)
model_official.eval()
print(type(model_official))


config = {
    "img_size": 384,
    "in_channels": 3,
    "patch_szie": 16,
    "embed_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4
}

model_vit = ViT(**config)
model_vit.eval()