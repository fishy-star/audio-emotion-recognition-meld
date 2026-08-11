import torch
import torch.nn as nn


class TrainableLayerWeighting(nn.Module):
    """Weighted sum across wav2vec2 layers, weights learned jointly with
    the downstream classifier (Pepino et al., 2021).

    Expects layer_outputs of shape (..., num_layers, hidden_dim) — the
    layer axis is always second-to-last, hidden_dim always last. This
    covers both a single utterance (num_layers, hidden_dim) and a batch
    (batch, num_layers, hidden_dim).
    """

    def __init__(self, num_layers=13):
        super().__init__()

        self.num_layers = num_layers

        # Initialized to zero -> softmax starts uniform (equivalent to
        # plain averaging across layers) until training pulls it apart.
        self.raw_weights = nn.Parameter(torch.zeros(num_layers))

    def get_weights(self):
        return torch.softmax(self.raw_weights, dim=0)

    def forward(self, layer_outputs):

        assert layer_outputs.shape[-2] == self.num_layers, (
            f"Expected {self.num_layers} layers at dim=-2, "
            f"got shape {tuple(layer_outputs.shape)}"
        )

        weights = self.get_weights()

        broadcast_shape = [1] * layer_outputs.dim()
        broadcast_shape[-2] = self.num_layers

        weights = weights.view(*broadcast_shape)

        return (weights * layer_outputs).sum(dim=-2)


if __name__ == "__main__":

    layer_weighting = TrainableLayerWeighting(num_layers=13)

    single_utterance = torch.randn(13, 768)
    single_out = layer_weighting(single_utterance)
    print("TrainableLayerWeighting — single utterance")
    print("  in: ", tuple(single_utterance.shape))
    print("  out:", tuple(single_out.shape))

    batch = torch.randn(4, 13, 768)
    batch_out = layer_weighting(batch)
    print("TrainableLayerWeighting — batch")
    print("  in: ", tuple(batch.shape))
    print("  out:", tuple(batch_out.shape))

    print("  learned weights sum to:", layer_weighting.get_weights().sum().item())
