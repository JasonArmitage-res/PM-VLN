"""
This file contains code from the following:
https://github.com/lil-lab/touchdown/blob/master/sdr/model.py
https://github.com/VegB/VLN-Transformer/blob/main/touchdown/models/RConcat.py
https://github.com/VegB/VLN-Transformer/blob/main/touchdown/main.py

The models below are used in the framework described in the following papers:
"TOUCHDOWN: Natural Language Navigation and Spatial Reasoning in Visual Street Environments"
https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_TOUCHDOWN_Natural_Language_Navigation_and_Spatial_Reasoning_in_Visual_Street_CVPR_2019_paper.pdf
"Multimodal Text Style Transfer for Outdoor Vision-and-Language Navigation"
https://arxiv.org/pdf/2007.00229.pdf

"""


import torch
from torch import nn
from utils.utils import padding_idx


class Conv_net(nn.Module):
    def __init__(self, opts):
        super(Conv_net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=4),
                nn.ReLU())
        self.fcl = nn.Linear(6 * 6 * 64, self.opts.hidden_dim)
        
    def forward(self, x):
        """
        :param x: [batch_size, 1, 100, 100]
        """
        x = self.conv(x)  # [batch_size, 64, 6, 6]
        x = x.view(-1, 6 * 6 * 64)  # [batch_size, 6 * 6 * 64]
        return self.fcl(x)  # [batch_size, 256]


class Text_linear(nn.Module):
    """
    Standard linear layer to generate sentence embeddings.
    """
    def __init__(self, opts):
        super(Text_linear, self).__init__()
        self.fcl = nn.Linear(768, opts.hidden_dim)

    def forward(self, x):
        """
        Forward through the model.
        :param x: (torch.tensor) Input representation (dim(x) = (batch, in_size)).
        :return: (torch.tensor) Return f(x) (dim(f(x) = (batch, out_size)).
        """
        return self.fcl(x)