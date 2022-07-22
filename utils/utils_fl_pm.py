"""
Source code for the PM-VLN module and FL_PM framework presented in our paper:
"A Priority Map for Vision-and-Language Navigation with Trajectory Plans and Feature-Location Cues"

This file contains code from the following:
https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_bert.html
https://github.com/maeotaku/pytorch_usm/blob/master/USM.py
https://github.com/maeotaku/pytorch_usm/blob/master/LoG.py
https://github.com/Duncanswilson/maxout-pytorch/blob/master/maxout_pytorch.ipynb

BERT is described in the following paper:
"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
https://aclanthology.org/N19-1423.pdf

The USM layer is described in the following paper:
"Unsharp Masking Layer: Injecting Prior Knowledge in Convolutional Networks for Image Classification"
https://dl.acm.org/doi/abs/10.1007/978-3-030-30508-6_1

Maxout units are described in this paper:
"Maxout Networks"
https://arxiv.org/pdf/1302.4389.pdf

"""


import torch
import torch.nn as nn
from packages.transformers_pm_vln import BertConfig
import torch.nn.functional as F
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Functions for BertEmbeddings.
"""
class BertEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, sequence_length=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

bert_config = BertConfig(
    vocab_size=30522,
    hidden_size=256,
    num_hidden_layers=1,
    num_attention_heads=1,
    num_labels=4,
)

# Instantiate BertEmbeddings 
bert_emb = BertEmbeddings(bert_config).to(device)


class SegEmb(nn.Module):
    """
    Generate embeddings for linguistic inputs.
    """
    def __init__(self):
        super(SegEmb, self).__init__()  
        self.embedding = bert_emb

    def forward(self, x_l):
        """
        Forward through the module.
        :param x_l: (torch.tensor) Linguistic inputs.
        :return x_out: (torch.tensor) Embeddings for linguistic inputs.
        """
        # Linguistic inputs
        x_out = self.embedding(x_l)
    
        return x_out


# Helper function to load checkpoint for the ConvNeXt Tiny model. 
def tpm_load_ckpt(ck_pt, optimizer):
    checkpoint = torch.load(ck_pt)
    cnxt_dict = checkpoint['state_dict']
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return cnxt_dict, optimizer, checkpoint['epoch']


"""
USM layer classes and component functions.
"""
def log2d(k, sigma, cuda=False):
    if cuda:
        ax = torch.round(torch.linspace(-math.floor(k/2), math.floor(k/2), k), out=torch.FloatTensor());
        ax = ax.cuda()
    else:
        ax = torch.round(torch.linspace(-math.floor(k/2), math.floor(k/2), k), out=torch.FloatTensor());
    y = ax.view(-1, 1).repeat(1, ax.size(0))
    x = ax.view(1, -1).repeat(ax.size(0), 1)
    x2 = torch.pow(x, 2)
    y2 = torch.pow(y, 2)
    s2 = torch.pow(sigma, 2)
    s4 = torch.pow(sigma, 4)
    hg = (-(x2 + y2)/(2.*s2)).exp()
    kernel_t = hg*(1.0 - (x2 + y2 / 2*s2)) * (1.0 / s4 * hg.sum())
    if cuda:
        kernel = kernel_t - kernel_t.sum() / torch.pow(torch.FloatTensor([k]).cuda(),2)
    else:
        kernel = kernel_t - kernel_t.sum() / torch.pow(torch.FloatTensor([k]),2)
    return kernel

class LoG2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, fixed_coeff=False, sigma=-1, stride=1, padding=0, dilation=1, cuda=False, requires_grad=True):
        super(LoG2d, self).__init__()
        self.fixed_coeff = fixed_coeff
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.cuda = cuda
        self.requires_grad = requires_grad
        if not self.fixed_coeff:
            if self.cuda:
                self.sigma = nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=self.requires_grad)
            else:
                self.sigma = nn.Parameter(torch.FloatTensor(1), requires_grad=self.requires_grad)
        else:
            if self.cuda:
                self.sigma = torch.cuda.FloatTensor([sigma])
            else:
                self.sigma = torch.FloatTensor([sigma])
            self.kernel = log2d(self.kernel_size, self.sigma, self.cuda)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        self.init_weights()

    def init_weights(self):
        if not self.fixed_coeff:
            self.sigma.data.uniform_(0.0001, 0.9999)

    def forward(self, input):
        batch_size, h, w = input.shape[0],  input.shape[2],  input.shape[3]
        if not self.fixed_coeff:
            self.kernel = log2d(self.kernel_size, self.sigma, self.cuda)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = self.kernel
        #kernel size is (out_channels, in_channels, h, w)
        output = F.conv2d(input.view(batch_size * self.in_channels, 1, h, w), kernel, padding=self.padding, groups=1)#, stride=self.stride, padding=self.padding, dilation=self.dilation)
        output = output.view(batch_size, self.in_channels, h, w)
        return output

class USMBase(LoG2d):
    def __init__(self, in_channels, kernel_size, fixed_coeff=False, sigma=-1, stride=1, dilation=1, cuda=False, requires_grad=True):
        #Padding must be forced so output size is = to input size
        #Thus, in_channels = out_channels
        padding = int((stride*(in_channels-1)+((kernel_size-1)*(dilation-1))+kernel_size-in_channels)/2)
        super(USMBase, self).__init__(in_channels, in_channels, kernel_size, fixed_coeff, sigma, stride, padding, dilation, cuda, requires_grad)
        self.alpha = None

    def i_weights(self):
        if self.requires_grad:
            super().init_weights()
            self.alpha.data.uniform_(0, 10)

    def assign_weight(self, alpha):
        if self.cuda:
            self.alpha = torch.cuda.FloatTensor([alpha])
        else:
            self.alpha = torch.FloatTensor([alpha])

    def forward(self, input):
        B = super().forward(input)
        U = input + self.alpha * B
        maxB = torch.max(torch.abs(B))
        maxInput = torch.max(input)
        U = U * maxInput/maxB
        return U

class AdaptiveUSM(USMBase):
    def __init__(self, in_channels, in_side, kernel_size, fixed_coeff=False, sigma=-1, stride=1, dilation=1, cuda=False, requires_grad=True):
        super(AdaptiveUSM, self).__init__(in_channels, kernel_size, fixed_coeff, sigma, stride, dilation, cuda, requires_grad)
        if self.requires_grad:
            if self.cuda:
                self.alpha = nn.Parameter(torch.cuda.FloatTensor(in_side, in_side), requires_grad=self.requires_grad)
            else:
                self.alpha = nn.Parameter(torch.FloatTensor(in_side, in_side), requires_grad=self.requires_grad)
            self.i_weights()


"""
Source code for maxout unit classes.
"""
class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

class maxout_mlp(nn.Module):
    def __init__(self, out_features, num_units=8):
        super(maxout_mlp, self).__init__()
        self.fc1_list = ListModule(self, "fc1_")
        self.fc2_list = ListModule(self, "fc2_")
        for _ in range(num_units):
            self.fc1_list.append(nn.Linear(256, 512))
            self.fc2_list.append(nn.Linear(512, out_features))

    def forward(self, x): 
        x = x.view(-1, 256).to('cuda:0')
        x = self.maxout(x, self.fc1_list).to('cuda:0')
        x = F.dropout(x, training=self.training).to('cuda:0')
        x = self.maxout(x, self.fc2_list).to('cuda:0')
        return F.log_softmax(x)

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x).to(device).to('cuda:0')
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x)).to('cuda:0')
        return max_output