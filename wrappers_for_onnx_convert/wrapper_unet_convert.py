import torch


class UNETWrapper(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module, transformer_options, extras):
        super().__init__()
        self.unet = unet
        self.transformer_options = transformer_options
        self.extras = extras

    def forward(self, x, timesteps, context, *args):
        extras = self.extras
        extra_args = {}
        for i in range(len(extras)):
            extra_args[extras[i]] = args[i]
        return self.unet(x, timesteps, context, transformer_options=self.transformer_options, **extra_args) 