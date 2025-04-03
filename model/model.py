import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ngpu=config['ngpu']
        self.layers=nn.ModuleList()
        self._build_layers(config['discriminator']['layers'])

    def _build_layers(self,layers_config):
        for layer_config in layers_config:
            layer_type=layer_config['type']
            layer_params=layer_config.get('params',{})
            print(layer_type)
            print(layer_params)
            if hasattr(nn,layer_type):
                layer=getattr(nn,layer_type)(**layer_params)
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            self.layers.append(layer)
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

class Generator(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ngpu=config['ngpu']
        self.layers=nn.ModuleList()
        self._build_layers(config['generator']['layers'])

    def _build_layers(self,layers_config):
        for layer_config in layers_config:
            layer_type=layer_config['type']
            params=layer_config.get('params',{})

            if hasattr(nn,layer_type):
                layer=getattr(nn,layer_type)(**params)
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            self.layers.append(layer)

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
# class DCGAN(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         generator=Generator(config)
#         discriminator=Discriminator(config)
