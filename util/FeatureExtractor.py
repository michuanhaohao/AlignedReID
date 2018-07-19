import torch.nn as nn
from IPython import embed

class FeatureExtractor(nn.Module):
    def __init__(self,submodule,extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "classfier":
                x = x.view(x.size(0),-1)
            if name is "base":
                for block_name, cnn_block in module._modules.items():
                    x = cnn_block(x)
                    if block_name in self.extracted_layers:
                        outputs.append(x)
        return outputs