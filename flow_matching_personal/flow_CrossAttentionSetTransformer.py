"""SetTransformer with Cross-Attention + FiLM (Flax / linen) + Flow integration

This file implements a SetTransformer variant that uses cross-attention
so that a latent parameter embedding (theta) queries the particle set.
Time embeddings (t_emb) are incorporated into both self-attention and
cross-attention (added to q/k projections similarly to how you used them),
AND a FiLM modulation conditioned on t_emb is applied inside the blocks.

This updated version REMOVES the per-particle return path and provides
an integrated `Flow` class that uses `SetTransformerCross` as the encoder.

Features:
- SAB: Self-Attention Block for sets (with optional time conditioning + FiLM)
- CrossAttentionBlock: queries = theta, keys/values = particle set (+ FiLM)
- SetTransformerCross: high-level module that produces a theta-aware summary
- Flow: Flax/diffusers-style Flow module that uses SetTransformerCross

Usage example:
    flow = Flow(sample_size=1000, dim_flow=7, N_dim=64, N_head=8, depth=3)
    # sample: (B, dim_flow)  (noisy / swapped in flow code)
    # encoder_hidden_states: (B, N, 6)  (phase-space)
    out = flow(timesteps, sample, encoder_hidden_states)

Return:
    out: (B, dim_flow)

"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

# diffusers / embedding utilities (kept for compatibility)
from diffusers.models.embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from diffusers.configuration_utils import ConfigMixin, flax_register_to_config
from diffusers.models.modeling_flax_utils import FlaxModelMixin

# conveniences
Dense = nn.Dense
DenseGeneral = nn.DenseGeneral
LayerNorm = nn.LayerNorm
from flax.linen import dot_product_attention


class MLPBlock(nn.Module):
    dim: int
    hidden_mult: int = 4

    @nn.compact
    def __call__(self, x):
        h = Dense(self.dim * self.hidden_mult)(x)
        h = nn.silu(h)
        h = Dense(self.dim)(h)
        return h


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on t_emb.

    Maps t_emb -> gamma, beta and applies gamma * x + beta.
    Keeps a small MLP to produce gamma/beta per channel.
    """
    out_dim: int
    hidden: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray):
        # x: (B, ..., C), t_emb: (B, D_t)
        C = self.out_dim
        # small MLP
        h = Dense(self.hidden)(t_emb)
        h = nn.silu(h)
        # final projection to 2*C
        # initialize final layer small so gamma ~ 1 and beta ~ 0 at start
        init = nn.initializers.normal(stddev=1e-3)
        params = Dense(2 * C, kernel_init=init)(h)  # (B, 2C)
        gamma, beta = jnp.split(params, 2, axis=-1)  # each (B, C)

        # broadcast to x shape
        if x.ndim == 3:
            # (B, N, C)
            gamma = gamma[:, None, :]
            beta = beta[:, None, :]
        elif x.ndim == 2:
            # (B, C)
            pass
        else:
            # support other shapes by broadcasting on the last axis
            expand_dims = [1] * (x.ndim - 2)
            gamma = gamma.reshape((gamma.shape[0],) + tuple(expand_dims) + (gamma.shape[-1],))
            beta = beta.reshape((beta.shape[0],) + tuple(expand_dims) + (beta.shape[-1],))

        return gamma * x + beta


class SAB(nn.Module):
    """Self-attention block for sets. Optionally uses time embedding and FiLM.

    Input: x (B, N, N_dim)
    t_emb: optional (B, D_t) - will be projected and broadcast to tokens
    """
    N_dim: int
    N_head: int
    ln: bool = False
    use_film: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: Optional[jnp.ndarray] = None):
        B, N, _ = x.shape
        dim_split = self.N_dim // self.N_head
        kernel_init = nn.initializers.variance_scaling(scale=1/3, mode="fan_in", distribution="uniform")

        # project q/k/v
        q = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(x)
        k = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(x)
        v = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(x)

        if t_emb is not None:
            # project t_emb to same head-split shape and broadcast
            t_proj_q = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(
                jnp.repeat(t_emb[:, None, :], N, axis=1)
            )
            t_proj_k = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(
                jnp.repeat(t_emb[:, None, :], N, axis=1)
            )
            q = q + t_proj_q
            k = k + t_proj_k

        a = dot_product_attention(q, k, v)  # (B, N, N_head, dim_split)
        out = a.reshape(B, N, self.N_dim)

        # residual
        out = out + x
        if self.ln:
            out = LayerNorm()(out)

        # FiLM modulation (apply before FFN)
        if self.use_film and t_emb is not None:
            out = FiLM(out_dim=self.N_dim)(out, t_emb)

        # FF
        out_ff = MLPBlock(dim=self.N_dim)(out)
        out = out + out_ff
        if self.ln:
            out = LayerNorm()(out)

        return out


class CrossAttentionBlock(nn.Module):
    """Cross-attention where queries = theta (latent params) and keys/values = set.

    - x (queries): (B, n_query, N_dim)  typically theta projected
    - context (keys/values): (B, N, N_dim)  particle embeddings
    - t_emb optional incorporated into q/k like in SAB

    Returns: (B, n_query, N_dim)
    """
    N_dim: int
    N_head: int
    ln: bool = False
    use_film: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: jnp.ndarray, t_emb: Optional[jnp.ndarray] = None):
        B, n_query, _ = x.shape
        N = context.shape[1]
        dim_split = self.N_dim // self.N_head
        kernel_init = nn.initializers.variance_scaling(scale=1/3, mode="fan_in", distribution="uniform")

        q = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(x)
        k = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(context)
        v = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(context)

        if t_emb is not None:
            # project t_emb to match query length and key length
            t_proj_q = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(
                jnp.repeat(t_emb[:, None, :], n_query, axis=1)
            )
            t_proj_k = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(
                jnp.repeat(t_emb[:, None, :], N, axis=1)
            )
            q = q + t_proj_q
            k = k + t_proj_k

        a = dot_product_attention(q, k, v)  # (B, n_query, N_head, dim_split)
        out = a.reshape(B, n_query, self.N_dim)

        out = out + x
        if self.ln:
            out = LayerNorm()(out)

        # FiLM modulation
        if self.use_film and t_emb is not None:
            out = FiLM(out_dim=self.N_dim)(out, t_emb)

        out_ff = MLPBlock(dim=self.N_dim)(out)
        out = out + out_ff
        if self.ln:
            out = LayerNorm()(out)

        return out


class SetTransformerCross(nn.Module):
    """SetTransformer variant that uses cross-attention for conditioning and FiLM.

    This variant REMOVES returning per-particle embeddings; it only returns
    the theta-aware summary (n_query outputs) and optionally projects to out_dim.

    Parameters
    ----------
    N_dim: int   -> model hidden dim
    N_head: int
    depth: int   -> number of SAB blocks
    n_query: int -> how many query tokens to create from theta (1 = single summary)
    out_dim: Optional[int] -> project theta_out to this dim (if provided)
    ln: bool
    use_film: bool

    Call: (x, t_emb=None, theta_emb=None)
    - x: (B, N, D_in)
    - t_emb: (B, D_t) optional
    - theta_emb: (B, D_theta) optional (if None, seeds are used as learned queries)

    Returns:
    theta_out: (B, n_query, out_dim or N_dim)  (if n_query==1, you might squeeze)
    """

    N_dim: int
    N_head: int
    depth: int = 2
    n_query: int = 1
    out_dim: Optional[int] = None
    ln: bool = False
    use_film: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: Optional[jnp.ndarray] = None, theta_emb: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # x: (B, N, D_in)
        B, N, _ = x.shape

        # 1) lift input tokens to model dim
        if x.shape[-1] != self.N_dim:
            x = Dense(self.N_dim)(x)  # (B, N, N_dim)

        # 2) feed through stacked SABs
        for _ in range(self.depth):
            x = SAB(N_dim=self.N_dim, N_head=self.N_head, ln=self.ln, use_film=self.use_film)(x, t_emb=t_emb)

        # 3) Prepare queries (theta). If theta_emb is provided, project it; otherwise learn seeds.
        if theta_emb is not None:
            # theta_emb: (B, D_theta) -> project to (B, n_query, N_dim)
            q = Dense(self.N_dim)(theta_emb)
            q = nn.silu(q)
            q = Dense(self.N_dim)(q)
            # expand to multiple queries if requested
            if self.n_query > 1:
                q = jnp.repeat(q[:, None, :], self.n_query, axis=1)  # (B, n_query, N_dim)
            else:
                q = q[:, None, :]
        else:
            # learned seeds
            seeds = self.param("seeds", nn.initializers.xavier_uniform(), (1, self.n_query, self.N_dim))
            q = jnp.repeat(seeds, B, axis=0)

        # 4) Cross-attention: queries attend to particle set
        q_out = CrossAttentionBlock(N_dim=self.N_dim, N_head=self.N_head, ln=self.ln, use_film=self.use_film)(q, x, t_emb=t_emb)

        # 5) optional projection
        if self.out_dim is not None:
            q_out = Dense(self.out_dim)(q_out)

        # If n_query==1, squeeze to (B, out_dim) for convenience
        if self.n_query == 1:
            q_out = jnp.squeeze(q_out, axis=1)

        return q_out


@flax_register_to_config
class Flow(nn.Module, FlaxModelMixin, ConfigMixin):
    """Flow module that uses SetTransformerCross as its encoder.

    Key args (exposed as dataclass config via flax_register_to_config):
      - sample_size: int (expected N particles when creating dummy inputs)
      - dim_flow: int (final output dim)
      - N_dim, N_head, depth: SetTransformer hyperparams
      - mean_pointcloud / std_pointcloud: arrays for normalization
    """
    sample_size: int = 1000
    dim_flow: int = 7
    N_dim: int = 64
    N_head: int = 8
    depth: int = 3
    ln: bool = True
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    use_film: bool = True

    # these paths/arrays can be set externally; provided here for compatibility
    mean_pointcloud = jnp.zeros((6,))
    std_pointcloud = jnp.ones((6,))

    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # initialize params by calling init with dummy inputs
        sample_shape = (1, self.sample_size, 6)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, self.dim_flow), dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.init(rngs, timesteps=timesteps, sample=encoder_hidden_states, encoder_hidden_states=sample)["params"]

    def setup(self) -> None:
        time_embed_dim = self.N_dim * 4

        # time
        self.time_proj = FlaxTimesteps(
            self.N_dim, flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
        )
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # simple param projection (instead of FlaxTimestepEmbedding)
        self.param_proj = nn.Sequential([
            Dense(self.N_dim),
            nn.silu,
            Dense(self.N_dim),
        ])

        # encoder
        self.SetTransformerEncoder = SetTransformerCross(
            N_dim=self.N_dim,
            N_head=self.N_head,
            depth=self.depth,
            n_query=1,
            out_dim=self.dim_flow,
            ln=self.ln,
            use_film=self.use_film,
        )

        self.dense_out = Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):
        # expects x shape (N, 6) or broadcastable
        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud + 1e-10)
        return normalized_x

    def __call__(
        self,
        timesteps: jnp.ndarray,
        sample: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        return_dict: bool = False,
        train: bool = False,
    ) -> jnp.ndarray:
        # timesteps: (B,) or scalar
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        # normalize particle cloud (encoder_hidden_states) and keep sample semantics
        # in your original Flow you swapped sample and encoder_hidden_states to match diffusers' API.
        encoder_hidden_states = jax.vmap(self.normalization, in_axes=0)(encoder_hidden_states)

        # time embedding
        timesteps = jnp.reshape(timesteps, -1)
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # swap: sample is expected to be the conditioning in diffusers' style here
        temp_ = sample
        sample = encoder_hidden_states   # now sample: (B, N, 6)
        encoder_hidden_states = temp_    # now encoder_hidden_states: (B, dim_flow)

        # project encoder_hidden_states (theta) to N_dim
        theta_emb = self.param_proj(encoder_hidden_states)  # (B, N_dim)

        # call SetTransformerCross: sample=(B,N,6), t_emb=(B, D_t), theta_emb=(B, N_dim)
        out = self.SetTransformerEncoder(sample, t_emb=t_emb, theta_emb=theta_emb)

        # out is (B, dim_flow) because SetTransformerCross out_dim=dim_flow and n_query=1
        return out


# === end of file ===
