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
    def __init__(self, encoder, decoder, num_centroids=256, num_speakers=None, speaker_embeddings_dim=0,
                 use_ma=False, ma_momentum=0.9):
        super(VQVAE, self).__init__()
        self.model_name = 'VQVAE'
        self.params_dict = {'num_centroids': num_centroids,
                            'num_speakers': num_speakers,
                            'speaker_embeddings_dim': speaker_embeddings_dim,
                            'use_ma': use_ma,
                            'ma_momentum': ma_momentum,
                            }
        self.encoder = encoder
        self.decoder = decoder
        self.quantize_fn = EuclideanQuantizer.apply
        self.num_centroids = num_centroids
        self.ma_momentum = ma_momentum
        centroids = torch.randn(self.num_centroids, self.encoder.output_dim)
        if use_ma:
            self.register_buffer('centroids', centroids)
            self.register_buffer('ma_num', centroids.clone())
            self.register_buffer('ma_denom', torch.zeros(self.num_centroids))
        else:
            self.centroids = torch.nn.Parameter(centroids, requires_grad=True)
            self.ma_num = None
            self.ma_denom = None
        self.num_speakers = num_speakers
        self.speaker_embeddings_dim = speaker_embeddings_dim
        self.use_ma = use_ma
        if self.num_speakers is not None:
            assert self.speaker_embeddings_dim
            self.speaker_embeddings = torch.nn.Embedding(self.num_speakers, self.speaker_embeddings_dim)

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

    def decode(self, x, speaker_ids=None):
        if self.num_speakers:
            assert speaker_ids is not None
            speaker_embedding = self.speaker_embeddings(speaker_ids).unsqueeze(-2)
            speaker_embedding = speaker_embedding.expand(speaker_embedding.size(0), x.size(1),
                                                         speaker_embedding.size(-1))
            x = torch.cat((x, speaker_embedding), dim=-1)
        x = self.decoder(x)
        return x

    def forward(self, x, speaker_ids=None):
        #  Assume x is of batch * length * dim
        x_enc = self.encoder(x)
        x_quant, ali = self.quantize(x_enc, return_alignment=True)
        ali_one_hot = torch.nn.functional.one_hot(ali, self.num_centroids)
        if self.training and self.use_ma:
            self.ma_denom = self.ma_momentum * self.ma_denom + (1 - self.ma_momentum) * ali_one_hot.sum(dim=0)
            ma_num_prop = torch.matmul(ali_one_hot.float().t(), x_enc.detach().view(-1, x_enc.size(-1)))
            self.ma_num = self.ma_momentum * self.ma_num + (1 - self.ma_momentum) * ma_num_prop
            self.centroids = self.ma_num / self.ma_denom.clamp_min(1)
        x_dec = self.decode(x_quant, speaker_ids=speaker_ids)
        return x_dec

    def save(self, outdir):
        with open(os.path.join(outdir, 'nnet_kind.txt'), 'w') as _kind:
            _kind.write(self.model_name)
        with open(os.path.join(outdir, 'nnet.json'), 'w') as _json:
            json.dump(self.params_dict, _json)
        for attr in ['encoder', 'decoder']:
            getattr(self, attr).save(os.path.join(outdir, attr))
        torch.save(self.state_dict(), os.path.join(outdir, 'nnet.mdl'))

    @classmethod
    def load_from_dir(cls, nnetdir, map_location=None):
        params_dict = {'num_centroids': 256,
                       'num_speakers': None,
                       'speaker_embeddings_dim': 0,
                       'use_ma': False,
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
        net = cls(encoder, decoder, **params_dict)
        net.to(map_location)
        state_dict = torch.load(os.path.join(nnetdir, 'nnet.mdl'),
                                map_location=map_location)
        net.load_state_dict(state_dict)
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
