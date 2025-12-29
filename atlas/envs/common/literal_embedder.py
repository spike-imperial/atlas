import chex
from flax import linen as nn

from .labeling_function import LabelingFunction


class LiteralEmbedder(nn.Module):
    @staticmethod
    def init_embedder(label_fn: LabelingFunction) -> "LiteralEmbedder":
        raise NotImplementedError


class BasicLiteralEmbedder(LiteralEmbedder):
    """
    A basic embedder that encodes literals independently of each other through
    an embedding matrix/module. Since Flax's embedding module encodes integers
    starting from 0, we sum the alphabet size (N) to the input literals, which
    results in negative literals mapped to 0...N-1, True mapped to N, and 
    positive literals mapped to N+1 and 2N.
    """
    alphabet_size: int
    d_feat: int

    def setup(self) -> None:
        # Since it embeds numbers in [0, n), we change the range of numbers we embed.
        self.embed = nn.Embed(
            num_embeddings=2 * self.alphabet_size + 1,
            features=self.d_feat,
        )

    def __call__(self, literal: chex.Array):
        return self.embed(literal + self.alphabet_size)

    @staticmethod
    def init_embedder(label_fn: LabelingFunction, d_feat: int = 64) -> "BasicLiteralEmbedder":
        return BasicLiteralEmbedder(
            alphabet_size=label_fn.get_alphabet_size(),
            d_feat=d_feat,
        )
