import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from typing import Tuple

# TODO: Discuss the AMP workaround issue implementation.
class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters [1].

    Args:
        dimensions:  number of spatial dimensions.
        num_embeddings: number of atomic elements in the codebook.
        embedding_dim: number of channels of the input and atomic elements.
        commitment_cost: scaling factor of the MSE loss between input and its quantized version.
            Defaults to 0.25 as per [1].
        decay: EMA decay. Defaults to 0.99 as per [1].
        epsilon: epsilon value. Defaults to 1e-5 as per [1].

    References:
        [1] Oord, A., Vinyals, O., and kavukcuoglu, k. 2017.
        Neural Discrete Representation Learning.
        In Advances in Neural Information Processing Systems (pp. 6306–6315).
        Curran Associates, Inc..
        https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148
        Commit 58d9a2746493717a7c9252938da7efa6006f3739
    """

    def __init__(
        self,
        dimensions: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        embedding_init: str = "normal",
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._dimensions = dimensions
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        if embedding_init == "normal":
            # Initialization is passed since the default one is normal inside the nn.Embedding
            pass
        elif embedding_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self._embedding.weight.data, mode="fan_in", nonlinearity="linear")
        self._embedding.weight.requires_grad = False

        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(self._num_embeddings))
        self.register_buffer("_ema_w", self._embedding.weight.data.clone())

        self._decay = decay
        self._epsilon = epsilon

        self._perplexity = torch.rand(1)

        # Precalculating required permutation shapes
        self._flatten_permutation = [0] + list(range(2, self._dimensions + 2)) + [1]
        self._quantization_permutation = [0, self._dimensions + 1] + list(range(1, self._dimensions + 1))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_input, encodings, encoding_indices = self.quantize(inputs)
        quantized = self.embed(encoding_indices)

        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                self._ema_cluster_size.data.mul_(self._decay).add_(torch.mul(encodings.sum(0), 1 - self._decay))

                # Laplace smoothing of the cluster size
                n = self._ema_cluster_size.sum()
                weights = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n

                dw = torch.mm(encodings.t(), flat_input)
                self._ema_w.data.mul_(self._decay).add_(torch.mul(dw, 1 - self._decay))

                self._embedding.weight.data.copy_(self._ema_w / weights.unsqueeze(1))

        # Encoding Loss
        loss = self._commitment_cost * F.mse_loss(quantized.detach(), inputs)

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity calculations
        avg_probs = (
            torch.histc(encoding_indices.float(), bins=self._num_embeddings, max=self._num_embeddings)
            .float()
            .div(encoding_indices.numel())
        )

        self._perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized

    def set_ema_decay(self, decay: float) -> None:
        self._decay = decay

    def get_ema_decay(self) -> float:
        return self._decay

    def set_commitment_cost(self, commitment_cost: float) -> None:
        self._commitment_cost = commitment_cost

    def get_commitment_cost(self) -> float:
        return self._commitment_cost

    def get_perplexity(self) -> torch.Tensor:
        return self._perplexity

    @amp.autocast(enabled=False)
    def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

        Args:
            inputs (torch.Tensor): Encoding space tensors

        Returns:
            torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
            torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self._num_embeddings].
            torch.Tensor: Quantization indices of shape [B,D,H,W,1]

        """
        encoding_indices_view = list(inputs.shape)
        del encoding_indices_view[1]

        inputs = inputs.float()

        # Converting to channel last format
        flat_input = inputs.permute(self._flatten_permutation).contiguous().view(-1, self._embedding_dim)

        # Calculate eucledian distances
        distances = (
            (flat_input ** 2).sum(dim=1, keepdim=True)
            + (self._embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
            - 2 * torch.mm(flat_input, self._embedding.weight.t())
        )

        # Mapping distances to indexes
        encoding_indices = torch.max(-distances, dim=1)[1]
        encodings = F.one_hot(encoding_indices, self._num_embeddings).float()

        # Quantize and reshape
        encoding_indices = encoding_indices.view(encoding_indices_view)

        return flat_input, encodings, encoding_indices

    @amp.autocast(enabled=False)
    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        """
        Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
        [B, D, H, W, self._embedding_dim] and reshapes them to [B, self._embedding_dim, D, H, W] to be fed to the
        decoder.

        Args:
            embedding_indices (torch.Tensor): Tensor in channel last format which holds indices referencing atomic
                elements from self._embedding

        Returns:
            torch.Tensor: Quantize space representation of encoding_indices in channel first format.
        """
        return self._embedding(embedding_indices).permute(self._quantization_permutation).contiguous()
