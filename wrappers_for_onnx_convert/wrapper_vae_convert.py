import torch


class VAEWrapper(torch.nn.Module):
    """Wrapper for VAE encoder/decoder to make it compatible with ONNX export"""
    def __init__(self, vae_model, is_encoder=False):
        super().__init__()
        self.vae = vae_model
        self.is_encoder = is_encoder

    def forward(self, x):
        if self.is_encoder:
            return self.vae.encode(x)
        else:
            return self.vae.decode(x) 