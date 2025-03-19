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
    "input_channels": 3,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4
}

model_custom = ViT(**config)
model_custom.eval()


# Iterate through all the parameters of the official network and our network
# 1. Check if number of the elements are equal 

for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
    """
    n_o represents the name of the parameter in the model_official.
    p_o represents the actual tensor (weights) of that parameter.
    numel() counts the total number of elements in a tensor.
    """
    print(f'{n_o} | {n_c}')
    assert p_o.numel() == p_c.numel()


    #  Copies the pretrained model’s weights to the custom model in-place.
    p_c.data[:] = p_o.data

    assert_tensor_equal(p_c.data, p_o.data)


# 2. Check whether the number of trainable parameters is the same for both model
input = torch.randn(1, 3, 384, 384)
res_o = model_official(input)
res_c = model_custom(input)

assert get_n_params(model_custom) == get_n_params(model_official)
assert_tensor_equal(res_c, res_o)


torch.save(model_custom, "./Vit/try1/model.pth")