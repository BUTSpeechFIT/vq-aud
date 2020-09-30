import os
import abc
import json
import torch
from torch import nn
import utils


class GenericLayer(abc.ABC, nn.Module):
    def __init__(self, previous_layer=None, input_dim=None, output_dim=None, samedim: bool = False,
                 **kwargs):
        super().__init__()
        self.model_name = 'GenericLayer'
        if input_dim is None:
            input_dim = previous_layer.get_output_dim()
        self.input_dim = input_dim
        if not isinstance(self.input_dim, int):
            raise ValueError(f'input_dim should be int got {type(self.input_dim)} instead')
        self.output_dim = output_dim
        if samedim:  # e.g for dropout and batchnorm
            self.output_dim = self.input_dim
        if not isinstance(self.output_dim, int):
            raise ValueError(f'output_dim should be int got {type(self.output_dim)} instead')

    def get_output_dim(self):
        return self.output_dim


class GRULayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None,
                 bidirectional: bool = False):
        super().__init__(previous_layer, input_dim, output_dim)
        self.bidirectional = bidirectional
        self.encoder = nn.GRU(self.input_dim, self.output_dim, num_layers=1, bidirectional=self.bidirectional,
                              batch_first=True)

    def forward(self, x: torch.Tensor, lengths=None, return_hidden: bool = False):
        x, hidden = self.encoder(x)
        if lengths is not None:
            x = x[torch.arange(x.size(0)), lengths]
        if return_hidden:
            return x, hidden
        else:
            return x

    def get_output_dim(self):
        ndir = 2 if self.bidirectional else 1
        return ndir * self.output_dim


class LSTMLayer(GRULayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None,
                 bidirectional: bool = False):
        super(GRULayer, self).__init__(previous_layer, input_dim, output_dim)
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(self.input_dim, self.output_dim, num_layers=1, bidirectional=self.bidirectional,
                               batch_first=True)


class FeedForwardLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None):
        super().__init__(previous_layer, input_dim, output_dim)
        self.output_dim = output_dim
        self.encoder = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x: torch.Tensor, lengths=None):
        x = self.encoder(x)
        if lengths is not None:
            x = x[torch.arange(x.size(0)), lengths]
        return x


# 1D CNN
class CNNLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None,
                 kind: str = '1D',
                 stride: int = 1,
                 bias=False,
                 kernel_size: int = 5, dilation: int = 1,
                 padding: float = None):
        super().__init__(previous_layer, input_dim, output_dim)
        # if not kernel_size % 2:
        #     kernel_size += 1
        self.kind = kind
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.bias = bias
        if padding is None:
            self.padding = int(dilation * (kernel_size - 1) / 2)
        else:
            self.padding = padding
        if self.kind == '1D':
            conv = nn.Conv1d
        elif self.kind == '2D':
            raise NotImplementedError(f'Conv2D')
            # conv = nn.Conv2d
        else:
            raise ValueError(f'expected convolutional type `1D`, got {self.kind} instead')
        self.encoder = conv(self.input_dim, self.output_dim,
                            kernel_size=self.kernel_size,
                            dilation=self.dilation,
                            padding=self.padding,
                            stride=self.stride,
                            bias=self.bias)

    def forward(self, x: torch.Tensor, lengths=None):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        if lengths is not None:
            x = x[torch.arange(x.size(0)), lengths]
        return x


class EmbeddingLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, output_dim: int = None,
                 pad=False, padding_idx=None):
        super().__init__(None, input_dim, output_dim)
        if pad and padding_idx is None:
            # In this case assume the last embedding is used for padding
            padding_idx = output_dim
        # In theory the line below should be within an `if pad:` block but is left outside for convenience so that
        # if `padding_idx` is set, it will behave as if `pad` is True regardless of the actual value of `pad`
        self.encoder = nn.Embedding(self.input_dim, self.output_dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class DropoutLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, p=0.):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.encoder = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class ReLULayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.encoder = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class BatchNormLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, **kwargs):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.encoder = nn.BatchNorm1d(self.input_dim, **kwargs)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return x.permute(0, 2, 1)


class LayerNormLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, affine=True, **kwargs):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.encoder = nn.LayerNorm(self.input_dim, elementwise_affine=affine, **kwargs)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        return x


class SubsamplingLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, factor=1, concat=False):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.subsampling_factor = factor
        self.concat = concat
        if self.concat:
            self.output_dim = self.subsampling_factor * self.output_dim
            self.encoder = lambda x: [x[:, i::self.subsampling_factor] for i in range(self.subsampling_factor)]
        else:
            self.encoder = lambda x: x[:, ::self.subsampling_factor]

    def forward(self, x: torch.Tensor):
        if self.concat:
            x = self.encoder(x)
            lens = [_.size(1) for _ in x]
            maxlen = max(lens)
            x = [torch.cat((arr, torch.zeros(x[0].size(0), maxlen - lv, x[0].size(-1))), dim=1)
                 for lv, arr in zip(lens, x)]
            x = torch.cat(x, dim=-1)
        else:
            x = self.encoder(x)
        return x


class UpsamplingLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 factor: int = 1, mode: str = 'nearest'):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.encoder = nn.Upsample(scale_factor=factor, mode=mode)

    def forward(self, x: torch.Tensor):
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = x.transpose(-1, -2).contiguous()
        return x


class JitterLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, p: float = 0):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.p = p

    def forward(self, x: torch.Tensor):
        if not self.training or self.p == 0:
            return x
        else:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            elif x.ndim != 3:
                raise ValueError(f'expected 2D or 3D tensor, got {x.ndim} instead')
            index_tensor = torch.arange(x.size(-2)).unsqueeze(0).unsqueeze(-1).expand_as(x)
            change_prob = torch.rand_like(index_tensor.float())
            index_change = (change_prob <= self.p).int()
            index_change = torch.where(change_prob > (self.p / 2), index_change, -index_change)  # t - 1
            index_tensor = index_tensor + index_change
            index_tensor.clamp_min_(0)
            index_tensor.clamp_max_(x.size(-2) - 1)
            x = torch.gather(x, -2, index_tensor).squeeze()
        return x


class CompositeModel(nn.Module):
    def __init__(self, layers_dict, ordering=None, input_dim=None):
        super().__init__()
        self.model_name = 'CompositeModel'
        if isinstance(layers_dict, str):
            with open(layers_dict) as _json:
                layers_dict = json.load(_json)
            ordering = layers_dict['ordering']
            layers_dict = layers_dict['layers']
        self.layers_dict = layers_dict
        if ordering is None:
            ordering = sorted([int(x) for x in layers_dict.keys()])
        self.ordering = [str(x) for x in ordering]
        if input_dim is not None:
            self.layers_dict[self.ordering[0]]['input_dim'] = input_dim
        layers = [make_layer(self.layers_dict[self.ordering[0]])]
        self.input_dim = layers[0].input_dim
        for key in self.ordering[1:]:
            layer = make_layer(self.layers_dict[key], previous_layer=layers[-1])
            layers.append(layer)
        self.output_dim = layers[-1].output_dim
        self.layers = nn.ModuleList(layers)
        print(self.layers)

    def forward_at_t(self, x, lengths=None):
        if lengths is None:
            lengths = [x.size(1)-1 for _ in x]
        return self(x, lengths)

    def summarize(self, input_sequence, mask=None, dim=1, **kwargs):
        x = self(input_sequence, **kwargs)
        if mask is not None:
            if mask.ndim < x.ndim:
                mask = mask.unsqueeze(-1).expand(*x.shape)
            x = x * mask
        return self.output_layer(x.sum(dim=dim))

    def forward(self, x, lengths=None, outputs_at_layer=None):
        intermediates = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if outputs_at_layer is not None and i in outputs_at_layer:
                intermediates.append(x)
        if lengths is not None:
            x = x[torch.arange(x.size(0)), lengths]
        if outputs_at_layer is not None:
            return x, intermediates
        return x

    def save(self, outdir):
        utils.chk_mkdir(outdir)
        with open(os.path.join(outdir, 'nnet.json'), 'w') as _json:
            savedict = {'layers': self.layers_dict,
                        'ordering': self.ordering}
            json.dump(savedict, _json)
        if hasattr(self, 'model_name'):
            with open(os.path.join(outdir, 'nnet_kind.txt'), 'w') as _kind:
                _kind.write(self.model_name)
        torch.save(self.state_dict(), os.path.join(outdir, 'nnet.mdl'))

    @classmethod
    def load_from_dir(cls, nnetdir, map_location=None):
        net = cls(os.path.join(nnetdir, 'nnet.json'))
        state_dict = torch.load(os.path.join(nnetdir, 'nnet.mdl'),
                                map_location=map_location)
        net.to(map_location)
        net.load_state_dict(state_dict)
        return net


_layers = {'GRULayer': GRULayer,
           'GRU': GRULayer,
           'LSTMLayer': LSTMLayer,
           'LSTM': LSTMLayer,
           'FeedForwardLayer': FeedForwardLayer,
           'FF': FeedForwardLayer,
           'CNNLayer': CNNLayer,
           'CNN': CNNLayer,
           'Dropout': DropoutLayer,
           'Drop': DropoutLayer,
           'BatchNormLayer': BatchNormLayer,
           'BatchNorm': BatchNormLayer,
           'LayerNormLayer': LayerNormLayer,
           'LayerNorm': LayerNormLayer,
           'ReLU': ReLULayer,
           'SubsamplingLayer': SubsamplingLayer,
           'Subs': SubsamplingLayer,
           'UpsamplingLayer': UpsamplingLayer,
           'Ups': UpsamplingLayer,
           'EmbeddingLayer': EmbeddingLayer,
           'Embedding': EmbeddingLayer,
           'JitterLayer': JitterLayer,
           'Jitter': JitterLayer,
           }


def make_layer(layer_dict: dict, previous_layer=None, **kwargs) -> GenericLayer:
    layer_dict = layer_dict.copy()
    layer_name = layer_dict.pop('layer_name')
    layer = _layers[layer_name](previous_layer=previous_layer, **layer_dict, **kwargs)
    return layer


def test_block():
    x0 = torch.randn(32, 100, 42)
    q0 = torch.randint(26, (32, 10))
    desc = {"0": {'layer_name': 'CNN', 'input_dim': 42, 'output_dim': 256},
            "6": {'layer_name': 'BatchNormLayer'},
            "9": {'layer_name': 'ReLU'},
            "4": {'layer_name': 'Drop', 'p': 0.5},
            "1": {'layer_name': 'GRU', 'output_dim': 256, 'bidirectional': True},
            "7": {'layer_name': 'BatchNormLayer'},
            "8": {'layer_name': 'Drop', 'p': 0.5},
            "3": {'layer_name': 'FF', 'output_dim': 20},
            "10": {'layer_name': 'SubsamplingLayer', 'factor': 2},
            "11": {'layer_name': 'SubsamplingLayer', 'factor': 3, 'concat': True},
            "12": {'layer_name': 'Embedding', 'input_dim': 26, 'output_dim': 32},
            "13": {'layer_name': 'CNN', 'output_dim': 256, 'kernel_size': 12},
            "14": {'layer_name': 'JitterLayer', 'p': 0.4},
            # "15": {'layer_name': 'CNN', 'output_dim': 256, 'kernel_size': 13, 'kind': '2D'},
            }
    model3 = CompositeModel(desc, ordering=['13', '10', '6', '9', '4', '1', '10', '7', '8', '3', '11',
                                            '14'], input_dim=42)
    qmod = CompositeModel(desc, ordering=['12', '1', '3'], input_dim=26)
    # model3.eval()

    y0 = model3(x0)

    model3.save('tmp/thisdir')
    model4 = CompositeModel.load_from_dir('tmp/thisdir')
    # model4.eval()

    y1 = model4(x0)
    qrep = qmod.forward_at_t(q0)
    print(x0.size())
    print(y1.size())
    print(qrep.size())
    print(abs((y0 - y1)).sum())
    inds = torch.randint(32, (32,))
    subs = 1
    for layer in model4.layers:
        try:
            subs *= layer.subsampling_factor
        except AttributeError:
            continue
    inds = inds // subs
    print(model4.forward_at_t(x0, inds).size())
    print(model4.input_dim)
    opt = torch.optim.Adam([{'params': model3.parameters()},
                            {'params': model4.parameters()}],
                           lr=1e-3)
    loss = y0.sum() + y1.sum()
    loss.backward()
    x, inter = model3(x0, outputs_at_layer=[10, 11])
    print((inter[0] == inter[1]).sum().float()/(x == x).sum().float())
    x.sum().backward()
    # print(model3.layers[0].encoder.weight.grad)
    # opt.param_groups[0].zero_grad()
    # print(model3.layers[0].encoder.weight.grad)


if __name__ == '__main__':
    test_block()
