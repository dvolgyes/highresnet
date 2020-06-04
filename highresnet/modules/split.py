#!/usr/bin/env python3
import torch
from torch.nn.functional import relu

# Let's replace ReLU with an immortal solution!
# Basic usage: instead of classic relu, use the split function,
# and in the next input double the number of input channels.
# In pseudo-code:
# ...
# x = relu(x)
# x = conv(in, out, ...)
# x = relu(x)
# x = conv(in, out, ...)
# x = relu(x)
#
# In new formalism:
# ...
# x = split(x)
# x = conv(in*2, out, ...)
# x = split(x)
# x = conv(in*2, out, ...)
# x = relu(x)
##  Remark: final activation might remain ReLU,
##  but often there is no activation after the last conv.



def split_relu_without_merge(t):
    zero = torch.tensor([0],dtype = t.dtype).to(t.device)
    smaller, larger = torch.min(t,zero), torch.max(t,zero)
    return smaller, larger

def split_relu(t):
    zero = torch.tensor([0],dtype = t.dtype).to(t.device)
    smaller, larger = torch.min(t,zero), torch.max(t,zero)
    return torch.cat((smaller, larger), dim=1)


class SplitReLU(torch.nn.Module):

    def __init__(self, merge = True):
        super().__init__()
        self.merge = merge

    def forward(self, x):
        if self.merge:
            return split_relu(x)
        else:
            return split_relu_without_merge(x)
