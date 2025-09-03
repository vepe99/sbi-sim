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


class MAB(nn.Module):
    N_dim: int
    N_head: int
    ln: bool = False

    @nn.compact
    def __call__(self, x, y, t_emb: Optional[jnp.ndarray] = None, theta_emb: Optional[jnp.ndarray] = None):
        """
        x: (B, N_x, D_in)    -> queries origin
        y: (B, N_y, D_in)    -> keys/values origin (may include extra tokens)
        t_emb: (B, D_t) or None
        theta_emb: (B, D_theta) or None  # will be concatenated as a token into y prior to projections
        """
        B = x.shape[0]
        N_x = x.shape[1]
        N_y = y.shape[1]

        # If theta_emb is provided, concatenate it as one extra token to y
        if theta_emb is not None:
            # project theta to N_dim first (so concatenation is compatible)
            theta_proj = nn.Dense(self.N_dim, kernel_init=nn.initializers.xavier_uniform())(theta_emb)  # (B, N_dim)
            theta_proj = theta_proj.squeeze(axis=1)
            theta_proj = theta_proj[:, None, :]  # (B, 1, N_dim)
            y = jnp.concatenate([theta_proj, y], axis=1)
            N_y = y.shape[1]

        dim_split = self.N_dim // self.N_head
        kernel_init = nn.initializers.variance_scaling(scale=1/3, mode="fan_in", distribution="uniform")

        # Project q/k/v with DenseGeneral -> (B, N, N_head, dim_split)
        q = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(x)
        k = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(y)
        v = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(y)

        # If time embedding is provided: project & broadcast and add to q and k (so time influences attention)
        if t_emb is not None:
            # project time to same head-split shape
            # create per-token t for x and per-key t for y
            t_proj_for_q = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(jnp.repeat(t_emb[:, None, :], N_x, axis=1))
            t_proj_for_k = DenseGeneral(features=(self.N_head, dim_split), kernel_init=kernel_init)(jnp.repeat(t_emb[:, None, :], N_y, axis=1))
            q = q + t_proj_for_q
            k = k + t_proj_for_k

        # dot-product attention: returns (B, N_x, N_head, dim_split)
        a = dot_product_attention(q, k, v)  # uses scaled dot-prod across heads

        # residual + reshape to (B, N_x, N_dim)
        o = (a + q).reshape(B, N_x, self.N_dim)

        if self.ln:
            o = LayerNorm()(o)

        # feed-forward (MLP) with residual
        o_ff = nn.activation.relu(DenseGeneral(self.N_dim, kernel_init=kernel_init)(o))
        o = o + o_ff

        if self.ln:
            o = LayerNorm()(o)

        return o


class SAB(nn.Module):
    N_dim: int
    N_head: int
    ln: bool = False

    @nn.compact
    def __call__(self, x, t_emb: Optional[jnp.ndarray] = None, theta_emb: Optional[jnp.ndarray] = None):
        # SAB is MAB(x, x) but we forward t_emb and theta_emb so these participate in attention
        return MAB(N_dim=self.N_dim, N_head=self.N_head, ln=self.ln)(x, x, t_emb=t_emb, theta_emb=theta_emb)

    
class PMA(nn.Module):
    """ Implementation of the 'Pooling by Multihead Attention'.

    Parameters
    ----------
    N_dim : int
        element-wise size of the output 

    N_head : int
        number of attention heads, must be a divisor of N_dim
        
    N_seed: int
        number of 'seed vectors'

    ln : bool
        if set to False, there is no layer normalization applied
        default: False
    """
    N_dim: int
    N_head: int
    N_seed: int
    ln: bool = False
        
    @nn.compact
    def __call__(self, x):
        N_batch = x.shape[0]
        s = self.param("seeds", nn.initializers.xavier_uniform(), (1,self.N_seed,self.N_dim))
        s = jnp.repeat(s, N_batch,axis=0).reshape((N_batch,self.N_seed,self.N_dim))
        
        return MAB(N_dim = self.N_dim,
                   N_head = self.N_head,
                   ln = self.ln) (s, x)
    
class SetTransformerEncoder(nn.Module):
    N_dim: int
    N_head: int
    depth: int = 2
    N_seed: int = 1
    out_dim: Optional[int] = None
    ln: bool = True

    @nn.compact
    def __call__(self, x, t_emb: Optional[jnp.ndarray] = None, theta_emb: Optional[jnp.ndarray] = None):
        # 1) lift input to model dimension
        if x.shape[-1] != self.N_dim:
            x = nn.Dense(self.N_dim)(x)

        # 2) optionally project t_emb
        if t_emb is not None:
            t_emb_proj = nn.Dense(self.N_dim)(t_emb)
            t_emb_proj = nn.silu(t_emb_proj)
            t_emb_proj = nn.Dense(self.N_dim)(t_emb_proj)
        else:
            t_emb_proj = None

        # 3) optionally project theta
        if theta_emb is not None:
            theta_emb = nn.Dense(self.N_dim)(theta_emb)
            theta_emb = nn.silu(theta_emb)
            theta_emb = nn.Dense(self.N_dim)(theta_emb)
        else:
            theta_emb = None

        # 4) pass through stacked SAB blocks
        for _ in range(self.depth):
            x = SAB(N_dim=self.N_dim, N_head=self.N_head, ln=self.ln)(
                x, t_emb=t_emb_proj, theta_emb=theta_emb
            )

        # 5) Pool with PMA
        x = PMA(N_dim=self.N_dim, N_head=self.N_head, N_seed=self.N_seed, ln=self.ln)(x)
        if self.N_seed == 1:
            x = jnp.squeeze(x, axis=1)

        # 6) optional final projection
        if self.out_dim is not None:
            x = nn.Dense(self.out_dim)(x)

        return x


@flax_register_to_config
class SetTransformer(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    A conditional 2D convolutional model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    Implementation based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition_flax.py

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for its generic methods
    implemented for all models (such as downloading or saving).

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.

    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        dim_flow (`int`, *optional*):
            The dimension of the flow.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            Enable memory efficient attention as described [here](https://arxiv.org/abs/2112.05682).
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """
    import_samples: int = 5 # or whatever type this should be
    sample_size: int = 1000
    dim_flow: int = 4  #must be set for the output dimension, like a final MLP output
    N_dim: int = 64
    N_head: int = 8
    depth: int = 4
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    addition_embed_type: Optional[str] = None
    addition_time_embed_dim: Optional[int] = None
    addition_embed_type_num_heads: int = 64
    projection_class_embeddings_input_dim: Optional[int] = None
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position/preprocess/mean_std_1e5_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position/preprocess/mean_std_1e5_pointcloud.npz')['std_x']

    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.sample_size, 6)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.N_dim), dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        added_cond_kwargs = None

        return self.init(rngs, sample, timesteps, encoder_hidden_states, added_cond_kwargs)["params"]

    def setup(self) -> None:
        time_embed_dim = self.N_dim * 4

        # time
        self.time_proj = FlaxTimesteps(
            self.N_dim , flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.config.freq_shift
        )
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # TODO add variable like time_embed_dim for params?
        self.param_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # addition embed types
        if self.addition_embed_type is None:
            self.add_embedding = None
        elif self.addition_embed_type == "text_time":
            if self.addition_time_embed_dim is None:
                raise ValueError(
                    f"addition_embed_type {self.addition_embed_type} requires `addition_time_embed_dim` to not be None"
                )
            self.add_time_proj = FlaxTimesteps(self.addition_time_embed_dim, self.flip_sin_to_cos, self.freq_shift)
            self.add_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)
        else:
            raise ValueError(f"addition_embed_type: {self.addition_embed_type} must be None or `text_time`.")

        self.SetTransformerEncoder = SetTransformerEncoder(
            N_dim=self.N_dim,
            N_head=self.N_head,
            depth=self.depth,
        )

        self.dense_out = nn.Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):

        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud + 1e-10)

        return normalized_x

    def __call__(
            self,
            timesteps: Union[jnp.ndarray, float, int],
            sample: jnp.ndarray,
            encoder_hidden_states: jnp.ndarray,
            added_cond_kwargs: Optional[Union[Dict, FrozenDict]] = None,
            down_block_additional_residuals: Optional[Tuple[jnp.ndarray, ...]] = None,
            mid_block_additional_residual: Optional[jnp.ndarray] = None,
            return_dict: bool = True,
            train: bool = False,
    ) -> jnp.ndarray:
        r"""
        Args:
            timesteps (`jnp.ndarray` or `float` or `int`): timesteps
            sample (`jnp.ndarray`): (batch, features) noisy inputs tensor
            encoder_hidden_states (`jnp.ndarray`): (batch_size, height, width) encoder hidden states
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            jnp.ndarray
        """
        # 1. time
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        print("compiling.....")
        # print(encoder_hidden_states.shape)
        encoder_hidden_states = jax.vmap(self.normalization, in_axes=0)(encoder_hidden_states)

        print("timesteps", timesteps.shape)
        timesteps = jnp.reshape(timesteps, -1)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        print("t_emb", t_emb.shape)

        # 1. swap sample (features) and encoder_hidden_states (conditioning for images)
        temp_ = sample
        sample = encoder_hidden_states
        encoder_hidden_states = temp_
        encoder_hidden_states = self.param_embedding(encoder_hidden_states)
        encoder_hidden_states = jnp.expand_dims(encoder_hidden_states, axis=1)

        sample = self.SetTransformerEncoder(sample, t_emb=t_emb, theta_emb=encoder_hidden_states)

        out = self.dense_out(sample, )

        return out



@flax_register_to_config
class SetTransformerPosition(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    A conditional 2D convolutional model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    Implementation based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition_flax.py

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for its generic methods
    implemented for all models (such as downloading or saving).

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.

    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        dim_flow (`int`, *optional*):
            The dimension of the flow.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            Enable memory efficient attention as described [here](https://arxiv.org/abs/2112.05682).
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """
    import_samples: int = 5 # or whatever type this should be
    sample_size: int = 1000
    dim_flow: int = 4  #must be set for the output dimension, like a final MLP output
    N_dim: int = 64
    N_head: int = 8
    depth: int = 4
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    addition_embed_type: Optional[str] = None
    addition_time_embed_dim: Optional[int] = None
    addition_embed_type_num_heads: int = 64
    projection_class_embeddings_input_dim: Optional[int] = None
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position/preprocess/mean_std_1e4_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position/preprocess/mean_std_1e4_pointcloud.npz')['std_x']

    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.sample_size, 6)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.N_dim), dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        added_cond_kwargs = None

        return self.init(rngs, sample, timesteps, encoder_hidden_states, added_cond_kwargs)["params"]

    def setup(self) -> None:
        time_embed_dim = self.N_dim * 4

        # time
        self.time_proj = FlaxTimesteps(
            self.N_dim , flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.config.freq_shift
        )
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # TODO add variable like time_embed_dim for params?
        self.param_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # addition embed types
        if self.addition_embed_type is None:
            self.add_embedding = None
        elif self.addition_embed_type == "text_time":
            if self.addition_time_embed_dim is None:
                raise ValueError(
                    f"addition_embed_type {self.addition_embed_type} requires `addition_time_embed_dim` to not be None"
                )
            self.add_time_proj = FlaxTimesteps(self.addition_time_embed_dim, self.flip_sin_to_cos, self.freq_shift)
            self.add_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)
        else:
            raise ValueError(f"addition_embed_type: {self.addition_embed_type} must be None or `text_time`.")

        self.SetTransformerEncoder = SetTransformerEncoder(
            N_dim=self.N_dim,
            N_head=self.N_head,
            depth=self.depth,
        )

        self.dense_out = nn.Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):

        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud + 1e-10)

        return normalized_x

    def __call__(
            self,
            timesteps: Union[jnp.ndarray, float, int],
            sample: jnp.ndarray,
            encoder_hidden_states: jnp.ndarray,
            added_cond_kwargs: Optional[Union[Dict, FrozenDict]] = None,
            down_block_additional_residuals: Optional[Tuple[jnp.ndarray, ...]] = None,
            mid_block_additional_residual: Optional[jnp.ndarray] = None,
            return_dict: bool = True,
            train: bool = False,
    ) -> jnp.ndarray:
        r"""
        Args:
            timesteps (`jnp.ndarray` or `float` or `int`): timesteps
            sample (`jnp.ndarray`): (batch, features) noisy inputs tensor
            encoder_hidden_states (`jnp.ndarray`): (batch_size, height, width) encoder hidden states
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            jnp.ndarray
        """
        # 1. time
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        print("compiling.....")
        # print(encoder_hidden_states.shape)
        encoder_hidden_states = jax.vmap(self.normalization, in_axes=0)(encoder_hidden_states)

        print("timesteps", timesteps.shape)
        timesteps = jnp.reshape(timesteps, -1)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        print("t_emb", t_emb.shape)

        # 1. swap sample (features) and encoder_hidden_states (conditioning for images)
        temp_ = sample
        sample = encoder_hidden_states
        encoder_hidden_states = temp_
        encoder_hidden_states = self.param_embedding(encoder_hidden_states)
        encoder_hidden_states = jnp.expand_dims(encoder_hidden_states, axis=1)

        sample = self.SetTransformerEncoder(sample, t_emb=t_emb, theta_emb=encoder_hidden_states)

        out = self.dense_out(sample, )

        return out



@flax_register_to_config
class SetTransformerPosition_1e5(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    A conditional 2D convolutional model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    Implementation based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition_flax.py

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for its generic methods
    implemented for all models (such as downloading or saving).

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.

    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        dim_flow (`int`, *optional*):
            The dimension of the flow.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            Enable memory efficient attention as described [here](https://arxiv.org/abs/2112.05682).
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """
    import_samples: int = 5 # or whatever type this should be
    sample_size: int = 1000
    dim_flow: int = 4  #must be set for the output dimension, like a final MLP output
    N_dim: int = 64
    N_head: int = 8
    depth: int = 4
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    addition_embed_type: Optional[str] = None
    addition_time_embed_dim: Optional[int] = None
    addition_embed_type_num_heads: int = 64
    projection_class_embeddings_input_dim: Optional[int] = None
    mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position/preprocess/mean_std_1e5_pointcloud.npz')['mean_x']
    std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position/preprocess/mean_std_1e5_pointcloud.npz')['std_x']

    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.sample_size, 6)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.N_dim), dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        added_cond_kwargs = None

        return self.init(rngs, sample, timesteps, encoder_hidden_states, added_cond_kwargs)["params"]

    def setup(self) -> None:
        time_embed_dim = self.N_dim * 4

        # time
        self.time_proj = FlaxTimesteps(
            self.N_dim , flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.config.freq_shift
        )
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # TODO add variable like time_embed_dim for params?
        self.param_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # addition embed types
        if self.addition_embed_type is None:
            self.add_embedding = None
        elif self.addition_embed_type == "text_time":
            if self.addition_time_embed_dim is None:
                raise ValueError(
                    f"addition_embed_type {self.addition_embed_type} requires `addition_time_embed_dim` to not be None"
                )
            self.add_time_proj = FlaxTimesteps(self.addition_time_embed_dim, self.flip_sin_to_cos, self.freq_shift)
            self.add_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)
        else:
            raise ValueError(f"addition_embed_type: {self.addition_embed_type} must be None or `text_time`.")

        self.SetTransformerEncoder = SetTransformerEncoder(
            N_dim=self.N_dim,
            N_head=self.N_head,
            depth=self.depth,
        )

        self.dense_out = nn.Dense(self.dim_flow, dtype=self.dtype)

    def normalization(self, x):

        normalized_x = (x - self.mean_pointcloud) / (self.std_pointcloud + 1e-10)

        return normalized_x

    def __call__(
            self,
            timesteps: Union[jnp.ndarray, float, int],
            sample: jnp.ndarray,
            encoder_hidden_states: jnp.ndarray,
            added_cond_kwargs: Optional[Union[Dict, FrozenDict]] = None,
            down_block_additional_residuals: Optional[Tuple[jnp.ndarray, ...]] = None,
            mid_block_additional_residual: Optional[jnp.ndarray] = None,
            return_dict: bool = True,
            train: bool = False,
    ) -> jnp.ndarray:
        r"""
        Args:
            timesteps (`jnp.ndarray` or `float` or `int`): timesteps
            sample (`jnp.ndarray`): (batch, features) noisy inputs tensor
            encoder_hidden_states (`jnp.ndarray`): (batch_size, height, width) encoder hidden states
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            jnp.ndarray
        """
        # 1. time
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        print("compiling.....")
        # print(encoder_hidden_states.shape)
        encoder_hidden_states = jax.vmap(self.normalization, in_axes=0)(encoder_hidden_states)

        print("timesteps", timesteps.shape)
        timesteps = jnp.reshape(timesteps, -1)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        print("t_emb", t_emb.shape)

        # 1. swap sample (features) and encoder_hidden_states (conditioning for images)
        temp_ = sample
        sample = encoder_hidden_states
        encoder_hidden_states = temp_
        encoder_hidden_states = self.param_embedding(encoder_hidden_states)
        encoder_hidden_states = jnp.expand_dims(encoder_hidden_states, axis=1)

        sample = self.SetTransformerEncoder(sample, t_emb=t_emb, theta_emb=encoder_hidden_states)

        out = self.dense_out(sample, )

        return out

