import os
import json
import torch
import layers
from layers import CompositeModel
from torch.autograd import Function


def pairwise_euclidean_distance(x, y):
    # Assume x is of l_x * d
    # and y is of l_y * d
    # output will be of l_x * l_y
    cross_term = -2 * x @ y.t()
    x_term = (x * x).sum(1).unsqueeze(1)
    y_term = (y * y).sum(1).unsqueeze(0)
    return cross_term + x_term + y_term


class EuclideanQuantizer(Function):
    @staticmethod
    def forward(ctx, input_tensor, centroids, alignment):
        # quantized = pairwise_euclidean_distance(input_tensor, centroids)
        return centroids[alignment]

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0], None, None


class GumbelSoftmaxQuantizer(Function):
    pass


class VQVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, num_centroids=256):
        super(VQVAE, self).__init__()
        self.model_name = 'VQVAE'
        self.params_dict = {'num_centroids': num_centroids}
        self.encoder = encoder
        self.decoder = decoder
        self.quantize_fn = EuclideanQuantizer.apply
        self.num_centroids = num_centroids
        centroids = torch.randn(self.num_centroids, self.encoder.output_dim, requires_grad=True)
        self.centroids = torch.nn.Parameter(centroids)

    def quantize(self, x, return_alignment=False):
        x_shape = x.shape
        x = x.contiguous()
        x = x.view(-1, x.size(-1))
        distances = pairwise_euclidean_distance(x, self.centroids)
        alignment = distances.argmin(dim=1)
        x = self.quantize_fn(x, self.centroids, alignment)
        x = x.view(*x_shape)
        if return_alignment:
            return x, alignment
        return x

    def forward(self, x):
        #  Assume x is of batch * length * dim
        x = self.encoder(x)
        x = self.quantize(x)
        x = self.decoder(x)
        return x

    def save(self, outdir):
        with open(os.path.join(outdir, 'nnet_kind.txt'), 'w') as _kind:
            _kind.write(self.model_name)
        with open(os.path.join(outdir, 'nnet.json'), 'w') as _json:
            json.dump(self.params_dict, _json)
        for attr in ['encoder', 'decoder']:
            getattr(self, attr).save(os.path.join(outdir, attr))

    @classmethod
    def load_from_dir(cls, nnetdir, map_location=None):
        params_dict = {'num_centroids': 256,
                       }
        with open(os.path.join(nnetdir, 'nnet.json')) as _json:
            params_dict_from_file = json.load(_json)
        params_dict.update(params_dict_from_file)

        encoder_model_name = open(os.path.join(nnetdir,
                                               'encoder',
                                               'nnet_kind.txt')).read().strip()
        encoder_kind = _model_names[encoder_model_name]
        encoder = encoder_kind.load_from_dir(os.path.join(nnetdir, 'encoder'),
                                             map_location=map_location)
        decoder_model_name = open(os.path.join(nnetdir,
                                               'decoder',
                                               'nnet_kind.txt')).read().strip()
        decoder_kind = _model_names[decoder_model_name]
        decoder = decoder_kind.load_from_dir(os.path.join(nnetdir, 'decoder'),
                                             map_location=map_location)
        net = cls(encoder, decoder, num_centroids=params_dict['num_centroids'])
        net.to(map_location)
        return net


_model_names = {'CompositeModel': layers.CompositeModel,
                'VQVAE': VQVAE,
                }


def get_model(model_name):
    return _model_names[model_name]


def test():
    from losses import MaskedMSELoss
    criterion = MaskedMSELoss()
    encoder = layers.GRULayer(input_dim=39, output_dim=500)
    decoder = layers.GRULayer(input_dim=500, output_dim=39)
    vq_vae = VQVAE(encoder, decoder, 3000).cuda()
    optimizer = torch.optim.Adam(vq_vae.parameters(),
                                 lr=1e-2
                                 )
    x = torch.randn(10, 1000, 39).cuda()
    gamma = 0.25
    for epoch in range(20):
        optimizer.zero_grad()
        encoded = vq_vae.encoder(x).contiguous()
        quantized, ali1 = vq_vae.quantize(encoded, return_alignment=True)
        y = vq_vae.decoder(quantized)
        # encoded = encoded.detach()
        # pairwise_distances = pairwise_euclidean_distance(encoded.view(-1, encoded.size(-1)), vq_vae.centroids)
        # ali0 = pairwise_distances.argmin(dim=0)
        # ali1 = pairwise_distances.argmin(dim=1)
        # loss0 = (encoded.view(-1, encoded.size(-1))[ali0] - vq_vae.centroids.detach()) ** 2
        # loss0 = loss0.sum() ** 0.5
        # loss0 = (encoded.view(-1, encoded.size(-1)) - vq_vae.centroids[ali1].detach()) ** 2
        predicted_centroids = vq_vae.centroids[ali1]
        encoded = encoded.view(-1, encoded.size(-1))
        loss0 = criterion(encoded, predicted_centroids.detach())
        loss1 = criterion(predicted_centroids, encoded.detach())
        rec_loss = criterion(y, x)
        # loss0 = loss0.sum()  # ** 0.5
        # loss1 = (encoded.view(-1, encoded.size(-1)).detach() - vq_vae.centroids[ali1]) ** 2
        # loss1 = loss1.sum()  # ** 0.5
        # y = vq_vae(x)
        # rec_loss = (x - y) ** 2
        # rec_loss = rec_loss.sum()  # ** 0.5
        total_loss = rec_loss + loss0 + loss1
        total_loss.backward()
        optimizer.step()
        print(f'iter: {epoch} - loss: {rec_loss.item()} - loss0: {loss0.item()} '
              f'- mean_centroid: {vq_vae.centroids.mean().item()}')


if __name__ == '__main__':
    test()
