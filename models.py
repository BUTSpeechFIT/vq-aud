import torch
import layers
from torch.autograd import Function


def pairwise_euclidean_distance(x, y):
    # Assume x is of l_x * d
    # and y is of l_y * d
    # output will be of l_x * l_y
    cross_term = -2 * x @ y.t()
    x_term = (x * x).sum(1).unsqueeze(1)
    y_term = (y * y).sum(1).unsqueeze(0)
    return (cross_term + x_term + y_term)


class EuclideanQuantizer(Function):
    @staticmethod
    def forward(ctx, input_tensor, centroids):
        quantized = pairwise_euclidean_distance(input_tensor, centroids)
        ctx.save_for_backward(quantized)
        return centroids[quantized.argmin(dim=1)]

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0], None


class GumbelSoftmaxQuantizer(Function):
    pass


class VQ_VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, num_centroids=256):
        super(VQ_VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantize_fn = EuclideanQuantizer.apply
        centroids = torch.randn(num_centroids, self.encoder.output_dim)
        self.centroids = torch.nn.Parameter(centroids)

    def quantize(self, x):
        x_shape = x.shape
        x = x.contiguous()
        x = x.view(-1, x.size(-1))
        x = self.quantize_fn(x, self.centroids)
        x = x.view(*x_shape)
        return x

    def forward(self, x):
        #  Assume x is of batch * length * dim
        x = self.encoder(x)
        x = self.quantize(x)
        x = self.decoder(x)
        return x


def test():
    encoder = layers.GRULayer(input_dim=39, output_dim=20)
    decoder = layers.GRULayer(input_dim=20, output_dim=39)
    vq_vae = VQ_VAE(encoder, decoder, 50)
    optimizer = torch.optim.Adam(vq_vae.parameters(),
                                 lr=1e-2
                                 )
    x = torch.randn(32, 100, 39)
    gamma = 0.25
    for epoch in range(20):
        optimizer.zero_grad()
        encoded = vq_vae.encoder(x).contiguous()
        quantized = vq_vae.quantize(encoded)
        y = vq_vae.decoder(quantized)
        # encoded = encoded.detach()
        pairwise_distances = pairwise_euclidean_distance(encoded.view(-1, encoded.size(-1)), vq_vae.centroids)
        # ali0 = pairwise_distances.argmin(dim=0)
        ali1 = pairwise_distances.argmin(dim=1)
        # loss0 = (encoded.view(-1, encoded.size(-1))[ali0] - vq_vae.centroids.detach()) ** 2
        # loss0 = loss0.sum() ** 0.5
        loss0 = (encoded.view(-1, encoded.size(-1)) - vq_vae.centroids[ali1].detach()) ** 2
        loss0 = loss0.sum()  # ** 0.5
        loss1 = (encoded.view(-1, encoded.size(-1)).detach() - vq_vae.centroids[ali1]) ** 2
        loss1 = loss1.sum()  # ** 0.5
        # y = vq_vae(x)
        rec_loss = (x - y) ** 2
        rec_loss = rec_loss.sum()  # ** 0.5
        total_loss = rec_loss + loss0 + loss1
        total_loss.backward()
        optimizer.step()
        print(f'iter: {epoch} - loss: {rec_loss.item()} - loss0: {loss0.item()} '
              f'- mean_centroid: {vq_vae.centroids.mean().item()}')


if __name__ == '__main__':
    test()
