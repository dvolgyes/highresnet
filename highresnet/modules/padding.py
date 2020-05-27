#!/usr/bin/env python3
import torch


def mixed_pad(input, pad, mode='constant', value=0, reversed_axes = False):
    """Mixed mode padding.

       :type input: tensor[B,C,D1,D2,...,DD]
       :type pad: int or tuple of ints with 2*D length
       :type mode: str or tuple
       :type value: float or tuple

       Dimension numbering: reverse order means [B,C,Dn,...,D2,D1],
       while regular order means [B,C,D1,D2,...Dn].

       If the mode is a string, everything falls back to classic padding.
       If the mode is a tuple, for each dimension the given padding is used.
       The input tensor must have D+2 dimensions (batch, channel, D1,D2,...)
       The padding must be integer or (left, right) padding for each dimension.
       If the padding value is not a single float, it must be a sequence
       with length of D always. E.g. the value '1' isn't used but required:
          mixed_pad(t, pad=(1,1,2,2,1,1),
                       mode=('circular','constant', 'constant'),
                       value = (1,2,3))

    """


    D = input.ndim - 2  # spatial dimensions (B,C,D1,D2,...)

    if not isinstance(pad, tuple):
        pad = (pad, pad) * D

    if not isinstance(mode, tuple):
        return torch.nn.functional.pad(input, pad, mode=mode, value=value)

    if not reversed_axes:
        pad = pad[::-1]
        mode = mode[::-1]

    assert len(mode) == D

    if not isinstance(value, tuple):
        value = (value,) * D

    zero = (0, 0)*D

    result = input

    for i in range(D):
        pre = zero[0:2*i]
        mid = pad[2*i:2*i+2]
        post = zero[2*i+2:]
        sequence = pre+mid+post
        if mode[i] == 'constant':
            result = torch.nn.functional.pad(result, sequence, mode='constant', value=value[i])
        else:
            result = torch.nn.functional.pad(result, sequence, mode=mode[i])
    return result


class MixedPad(torch.nn.Module):

    def __init__(self, pad, mode='constant', value=0, reversed_axes = False):
        super().__init__()

        self.pad = pad
        self.mode = mode
        self.value = value
        self.reversed_axes = reversed_axes

    def forward(self, x):
        return mixed_pad(x, self.pad, self.mode, self.value, reversed_axes)
