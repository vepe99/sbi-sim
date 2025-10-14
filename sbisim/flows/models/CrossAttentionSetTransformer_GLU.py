from typing import Dict, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from diffusers.configuration_utils import ConfigMixin, flax_register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from diffusers.models.modeling_flax_utils import FlaxModelMixin
from diffusers.models.unets.unet_2d_blocks_flax import (
    FlaxCrossAttnDownBlock2D,
    FlaxCrossAttnUpBlock2D,
    FlaxDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
    FlaxUpBlock2D,
)
from ..cnf import ContinuousNormalizingFlow

from flax.linen import dot_product_attention, DenseGeneral, LayerNorm
import flax.linen as nn
import jax.numpy as jnp
from typing import Any, Callable, Sequence, Optional, Union


from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

# diffusers / embedding utilities (kept for compatibility)
from diffusers.models.embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from diffusers.configuration_utils import ConfigMixin, flax_register_to_config
from diffusers.models.modeling_flax_utils import FlaxModelMixin
kernel_init_fn = nn.initializers.glorot_normal
bias_init_fn = nn.initializers.normal

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


class GLU(nn.Module):

    N_dim: int

    def setup(self):

        self.dense1 = nn.Dense(self.N_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())
        self.dense2 = nn.Dense(self.N_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())

    def __call__(self, x, cond):

        x = self.dense1(x)
        y = self.dense2(cond)

        return x * nn.sigmoid(y)


class SAB(nn.Module):
    """Self-attention block for sets. Optionally uses time embedding and FiLM.

    Input: x (B, N, N_dim)
    t_emb: optional (B, D_t) - will be projected and broadcast to tokens
    """
    N_dim: int
    N_head: int
    ln: bool = False


    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: Optional[jnp.ndarray] = None):
        B, N, _ = x.shape
        dim_split = self.N_dim // self.N_head
        kernel_init = nn.initializers.variance_scaling(scale=1/3, mode="fan_in", distribution="uniform")

        # project q/k/v
        q = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(x)
        k = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(x)
        v = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(x)


        a = dot_product_attention(q, k, v)  # (B, N, N_head, dim_split)
        out = a.reshape(B, N, self.N_dim)

        # residual
        out = out + x
        if self.ln:
            out = LayerNorm()(out)

        out = GLU(N_dim=self.N_dim)(out, t_emb[:, None, :])

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

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: jnp.ndarray, t_emb: Optional[jnp.ndarray] = None):
        B, n_query, _ = x.shape
        N = context.shape[1]
        dim_split = self.N_dim // self.N_head
        kernel_init = nn.initializers.variance_scaling(scale=1/3, mode="fan_in", distribution="uniform")

        q = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(x)
        k = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(context)
        v = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(context)


        a = dot_product_attention(q, k, v)  # (B, n_query, N_head, dim_split)
        out = a.reshape(B, n_query, self.N_dim)

        out = out + x

        # Broadcast or pooled conditioning for shape safety
        context_pooled = jnp.mean(context, axis=1, keepdims=True)  # (B, 1, N_dim)
        cond = jnp.concatenate([t_emb[:, None, :], context_pooled], axis=-1)  # (B, 1, D_t + N_dim)
        cond = jnp.repeat(cond, out.shape[1], axis=1)  # match query tokens
        out = GLU(N_dim=self.N_dim)(out, cond)


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

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: Optional[jnp.ndarray] = None, theta_emb: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # x: (B, N, D_in)
        B, N, _ = x.shape

        # 1) lift input tokens to model dim
        if x.shape[-1] != self.N_dim:
            x = Dense(self.N_dim)(x)  # (B, N, N_dim)

        # 2) feed through stacked SABs
        for _ in range(self.depth):
            x = SAB(N_dim=self.N_dim, N_head=self.N_head, ln=self.ln, )(x, t_emb=t_emb)

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
        for _ in range(self.depth):
            q = CrossAttentionBlock(N_dim=self.N_dim, N_head=self.N_head, ln=self.ln, )(q, x, t_emb=t_emb)

        # 5) optional projection
        if self.out_dim is not None:
            q_out = Dense(self.out_dim)(q)

        # If n_query==1, squeeze to (B, out_dim) for convenience
        if self.n_query == 1:
            q_out = jnp.squeeze(q_out, axis=1)

        return q_out


@flax_register_to_config
class SetTransformerPosition_newprior(nn.Module, FlaxModelMixin, ConfigMixin):
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

    # these paths/arrays can be set externally; provided here for compatibility
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_newprior/preprocess/mean_std_1e5_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_newprior/preprocess/mean_std_1e5_pointcloud.npz')['std_x']

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
        )

        self.dense_out = Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):
        # expects x shape (N, 6) or broadcastable
        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud)
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



@flax_register_to_config
class SetTransformerPosition_uniformprior(nn.Module, FlaxModelMixin, ConfigMixin):
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

    # these paths/arrays can be set externally; provided here for compatibility
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior/preprocess/mean_std_2e5_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior/preprocess/mean_std_2e5_pointcloud.npz')['std_x']

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

        )

        self.dense_out = Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):
        # expects x shape (N, 6) or broadcastable
        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud)
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

@flax_register_to_config
class SetTransformerPosition_uniformprior_30e5(nn.Module, FlaxModelMixin, ConfigMixin):
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

    # these paths/arrays can be set externally; provided here for compatibility
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior/preprocess/mean_std_3e5_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior/preprocess/mean_std_3e5_pointcloud.npz')['std_x']

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
        )

        self.dense_out = Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):
        # expects x shape (N, 6) or broadcastable
        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud)
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
    

@flax_register_to_config
class SetTransformerPosition_uniformprior_TSIT5(nn.Module, FlaxModelMixin, ConfigMixin):
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


    # these paths/arrays can be set externally; provided here for compatibility
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior_TSIT5/preprocess/mean_std_2e5_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior_TSIT5/preprocess/mean_std_2e5_pointcloud.npz')['std_x']

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
        )

        self.dense_out = Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):
        # expects x shape (N, 6) or broadcastable
        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud)
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


@flax_register_to_config
class SetTransformerPosition_uniformprior_error(nn.Module, FlaxModelMixin, ConfigMixin):
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

    # these paths/arrays can be set externally; provided here for compatibility
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior/preprocess/mean_std_2e5_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior/preprocess/mean_std_2e5_pointcloud.npz')['std_x']

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
        )

        self.dense_out = Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):
        # expects x shape (N, 6) or broadcastable
        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud)
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
        # encoder_hidden_states shape: (batch_size, 1000, 6)
        error_rng = self.make_rng('params')
        noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 1e-3])  # Shape: (6,)
        
        # Generate noise with correct shape: (batch_size, 1000, 6)
        noise_shape = encoder_hidden_states.shape
        noise = jax.random.normal(error_rng, noise_shape)  # (batch_size, 1000, 6)
        
        # Apply noise with correct broadcasting: each of the 6 dimensions gets its own noise std
        encoder_hidden_states = encoder_hidden_states + noise * noise_std[None, None, :]
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
    



@flax_register_to_config
class SetTransformerPosition_fixposition_uniformprior_TSIT5(nn.Module, FlaxModelMixin, ConfigMixin):
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

    # these paths/arrays can be set externally; provided here for compatibility
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position_uniform_prior_TSTIT5/preprocess/mean_std_1e5_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position_uniform_prior_TSTIT5/preprocess/mean_std_1e5_pointcloud.npz')['std_x']

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
        time_embed_dim = self.N_dim 

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
        )

        self.dense_out = Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):
        # expects x shape (N, 6) or broadcastable
        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud)
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
        # encoder_hidden_states shape: (batch_size, 1000, 6)

        
        # Apply noise with correct broadcasting: each of the 6 dimensions gets its own noise std
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


@flax_register_to_config
class SetTransformerPositionPositions_fixedtime_uniformprior_TSIT5(nn.Module, FlaxModelMixin, ConfigMixin):
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

    # these paths/arrays can be set externally; provided here for compatibility
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_fixed_time_uniform_prior_TSTIT5/preprocess/mean_std_1e5_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_fixed_time_uniform_prior_TSTIT5/preprocess/mean_std_1e5_pointcloud.npz')['std_x']

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
        )

        self.dense_out = Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):
        # expects x shape (N, 6) or broadcastable
        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud)
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
        # encoder_hidden_states shape: (batch_size, 1000, 6)

        
        # Apply noise with correct broadcasting: each of the 6 dimensions gets its own noise std
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
    

