from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    batch_size, batch_length = seq.shape[:2]
    subsequent_mask = (
        1 - torch.triu(torch.ones((1, batch_length, batch_length), device=seq.device), diagonal=1)
    ).bool()
    return subsequent_mask


def get_subsequent_mask_with_batch_length(batch_length, device):
    """For masking out the subsequent info."""
    subsequent_mask = (1 - torch.triu(torch.ones((1, batch_length, batch_length), device=device), diagonal=1)).bool()
    return subsequent_mask


def get_vector_mask(batch_length, device):
    mask = torch.ones((1, 1, batch_length), device=device).bool()
    return mask


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_heads: int = 8, d_model: int = 512, attn_dropout_p: float = 0.0, proj_dropout_p: float = 0.1):
        """
        Args:
            n_heads: number of heads
            d_model: dimension of the model
            attn_dropout_p: dropout probability for attention weights
            proj_dropout_p: dropout probability for the output tensor
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"`d_model` ({d_model}) must be divisible by `n_heads` ({n_heads})")

        self.n_heads = n_heads
        self.d_model = d_model
        self.attn_dropout_p = attn_dropout_p
        self.proj_dropout_p = proj_dropout_p
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.proj_dropout = nn.Dropout(proj_dropout_p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, C = x.shape

        qkv = self.qkv(x)  # B x T x 3C
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)  # 3 x B x n_heads x T x d_head
        q, k, v = qkv.unbind(0)  # B x n_heads x T x d_head

        x = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=mask is None,
            )  # B x n_heads x T x d_head
            .transpose(1, 2)  # B x T x n_heads x d_head
            .reshape(B, T, C)
        )

        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout_p=0.1, activation: Type[torch.nn.Module] = nn.ReLU):
        """
        Args:
            d_in: the dimension of the input tensor
            d_hid: the dimension of the hidden layer
            dropout_p: dropout probability
            activation: activation function
        """
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = activation()

    def forward(self, x):
        x = self.w_2(self.activation(self.w_1(x)))
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        mlp_factor: int = 2,
        n_heads: int = 8,
        attn_dropout_p: float = 0.0,
        proj_dropout_p=0.1,
        ffn_dropout_p: float = 0.1,
        ffn_activation: Type[torch.nn.Module] = nn.ReLU,
        norm_layer: Type[torch.nn.Module] = nn.LayerNorm,
        **norm_kwargs,
    ):
        """
        Args:
            d_model: dimension of the model
            mlp_factor: the multiplier of the hidden layer dimension in the pointwise feed-forward network
            n_heads: number of attention heads
            attn_dropout_p: dropout probability for attention weights
            proj_dropout_p: dropout probability for the MHSA output tensor
            ffn_dropout_p: dropout probability for the pointwise feed-forward network
            ffn_activation: activation function for the pointwise feed-forward network
            norm_layer: normalization layer
            norm_kwargs: keyword arguments for the normalization layer
        """
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(
            n_heads=n_heads,
            d_model=d_model,
            attn_dropout_p=attn_dropout_p,
            proj_dropout_p=proj_dropout_p,
        )
        self.ffn = PositionwiseFeedForward(
            d_in=d_model,
            d_hid=d_model * mlp_factor,
            dropout_p=ffn_dropout_p,
            activation=ffn_activation,
        )
        self.norm1 = norm_layer(d_model, **norm_kwargs)
        self.norm2 = norm_layer(d_model, **norm_kwargs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.mhsa(x, mask=mask))
        x = self.norm2(x + self.ffn(x))
        return x


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_length: int, d_embd: int):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = d_embd
        self.pos_emb = nn.Embedding(self.max_length, d_embd)
        self.range = torch.arange(max_length)

    def forward(self, feat):
        pos_emb = self.pos_emb(self.range.to(feat.device))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])
        feat = feat + pos_emb[:, : feat.shape[1], :]
        return feat


class StochasticTransformer(nn.Module):
    def __init__(
        self,
        d_stoch: int,
        d_action: int,
        d_model: int = 512,
        mlp_factor: int = 2,
        n_layers: int = 2,
        n_heads: int = 8,
        max_length: int = 16,
        attn_dropout_p: float = 0.0,
        proj_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        block_norm_layer: Type[torch.nn.Module] = nn.LayerNorm,
        **block_norm_norm_kwargs,
    ):
        """
        Args:
            d_stoch: dimension of the stochastic latent variable. Default is 32*32=1024
            d_action: dimension of the discrete action space
            d_model: dimension of the model
            mlp_factor: the multiplier of the hidden layer dimension in the pointwise feed-forward network
            n_layers: number of layers
            n_heads: number of attention heads
            max_length: the maximum length of the sequence
            attn_dropout_p: dropout probability for attention weights
            proj_dropout_p: dropout probability for the output tensor in the MHSA layer
            ffn_dropout_p: dropout probability for the pointwise feed-forward network
            block_norm_layer: normalization layer for the transformer block
            block_norm_norm_kwargs: keyword arguments for the normalization layer
        """
        super().__init__()
        self.d_stoch = d_stoch
        self.d_action = d_action
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.action_mixer = nn.Sequential(
            nn.Linear(d_stoch + d_action, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, d_embd=d_model)
        self.transformer = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    mlp_factor=mlp_factor,
                    n_heads=n_heads,
                    attn_dropout_p=attn_dropout_p,
                    proj_dropout_p=proj_dropout_p,
                    ffn_dropout_p=ffn_dropout_p,
                    norm_layer=block_norm_layer,
                    **block_norm_norm_kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, d_stoch)

    def forward(self, samples: torch.Tensor, action: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.action_mixer(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)

        for block in self.transformer:
            feats = block(feats, mask)

        feats = self.head(feats)
        return feats


if __name__ == "__main__":
    device = "cuda"
    model = StochasticTransformer(d_stoch=16, d_action=4, d_model=32, max_length=16).to(device)
    samples = torch.randn(4, 16, 16, device=device)
    action = F.one_hot(torch.randint(0, 4, (4, 16), device=device), num_classes=4).float()
    mask = get_subsequent_mask_with_batch_length(16, device=device)
    out = model(samples, action, None)
    print(out.shape)
