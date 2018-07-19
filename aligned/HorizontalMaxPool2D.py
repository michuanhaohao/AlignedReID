import torch.nn as nn

class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()


    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))