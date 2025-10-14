from dataclasses import field
from typing import List, Tuple, Optional, Any, Sequence, Union

from jax import Array, dtypes, random
import jax
from jax._src.nn.initializers import RealNumeric, DTypeLikeInexact, Initializer, _compute_fans, lecun_uniform
from jax._src import core

from diffusers.models.embeddings_flax import FlaxTimesteps
from ..cnf import ContinuousNormalizingFlow
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax

import flax.linen as nn

kernel_init_fn = nn.initializers.glorot_normal
bias_init_fn = nn.initializers.normal

def uniform_shifted(scale: RealNumeric = 1e-3,
            dtype: DTypeLikeInexact = jnp.float_) -> Initializer:

  def init(key,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.uniform(key, shape, dtype) * 2 * jnp.array(scale, dtype) - jnp.array(scale, dtype)
  return init

def pytorch_kernel_init(in_axis: Sequence[int] = -2,
        out_axis: Sequence[int] = -1,
        batch_axis: Sequence[int] = (),
        dtype: DTypeLikeInexact = jnp.float_) -> Initializer:

      def init(key,
              shape: core.Shape,
              dtype: DTypeLikeInexact = dtype) -> Array:
             dtype = dtypes.canonicalize_dtype(dtype)
             shape = core.canonicalize_shape(shape)
             dtype = dtypes.canonicalize_dtype(dtype)
             fan_in, fan_out = _compute_fans(shape, in_axis, out_axis, batch_axis)
             return (random.uniform(key, shape, dtype) * 2 * jnp.sqrt(jnp.array(1 / fan_in, dtype))
                     - jnp.sqrt(jnp.array(1 / fan_in, dtype)))
      return init

def pytorch_bias_init(
        in_features: int,
        dtype: DTypeLikeInexact = jnp.float_) -> Initializer:

    def init(key,
            shape: core.Shape,
            dtype: DTypeLikeInexact = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return (random.uniform(key, shape, dtype) * 2 * jnp.sqrt(jnp.array(1 / in_features, dtype))
                - jnp.sqrt(jnp.array(1 / in_features, dtype)))

    return init



class GLU(nn.Module):

    dim: int

    def setup(self):

        self.dense1 = nn.Dense(self.dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())
        self.dense2 = nn.Dense(self.dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())

    def __call__(self, x, t, theta):

        x = self.dense1(x)
        y = self.dense2(jnp.concatenate([t, theta], axis=-1))

        return x * nn.sigmoid(y)

class BaseFMPE(nn.Module):

    
    gelu: List[bool]
    time_embed_dim: int
    dim_flow: int
    residual_blocks = [(64, 3), (32, 2)]

    def setup(self):

        self.time_proj = FlaxTimesteps(
            self.time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )

        layers = []

        self.upsample = nn.Dense(self.residual_blocks[0][0],
                                 kernel_init=kernel_init_fn(), use_bias=False)
        print(self.residual_blocks)

        for i, (out_dim, repetitions) in enumerate(self.residual_blocks):

            for j in range(repetitions):

                # Residual block
                layers.append(nn.Dense(out_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()))
                layers.append(nn.Dense(out_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()))

                if self.gelu[i]:
                    layers.append(GLU(out_dim))

            if i < len(self.residual_blocks) - 1:

                layers.append(nn.Dense(self.residual_blocks[i+1][0], kernel_init=kernel_init_fn(),
                                       bias_init=bias_init_fn()))


        layers.append(nn.Dense(self.dim_flow, kernel_init=kernel_init_fn(), use_bias=False))

        self.layers = layers

    def __call__(self,
            timesteps: Union[jnp.ndarray, float, int],
            theta: jnp.ndarray,
            y: jnp.ndarray,
            train: bool = False,):

        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 1:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 1)

        if len(timesteps.shape) == 1:
            timesteps = jnp.repeat(timesteps, y.shape[0], axis=0)

        t_emb = self.time_proj(timesteps[:, 0])
        t_emb = jnp.concatenate([t_emb, timesteps[:, 1:]], axis=-1)

        y = jnp.reshape(y, (y.shape[0], -1))

        if y.shape[1] == 0:
            y = jnp.zeros((y.shape[0], 1))

        y = self.upsample(y)

        c = 0

        for i, (out_dim, repetitions) in enumerate(self.residual_blocks):

                for j in range(repetitions):

                    y_ = y
                    y = self.layers[c](y)
                    y = nn.activation.elu(y)
                    y = self.layers[c+1](y)
                    y = nn.activation.elu(y)
                    y += y_

                    c += 2

                    if self.gelu[i]:
                        y = self.layers[c](y, t_emb, theta)
                        c += 1

                if i < len(self.residual_blocks) - 1:
                    y = self.layers[c](y)
                    c += 1

        y = self.layers[-1](y)
        return y

class FMPE(ContinuousNormalizingFlow):

    residual_blocks: List[Tuple[int, int]] = field(default_factory=list)
    gelu: List[bool] = field(default_factory=list)
    time_embed_dim: int = 16

    def setup(self):
        super().setup()

        self.model = BaseFMPE(
            residual_blocks=self.residual_blocks,
            gelu=self.gelu,
            time_embed_dim=self.time_embed_dim,
            dim_flow=self.dim_flow
        )

class BaseBenchmarkFMPE(nn.Module):

    hidden_dim: int
    num_blocks: int
    out_dim: int
    activation_fn: str = 'elu'

    def setup(self):

        blocks = []

        if self.activation_fn == 'elu':
            self.activation = nn.elu
        elif self.activation_fn == 'gelu':
            self.activation = nn.gelu
        else:
            raise ValueError(f"Activation function {self.activation_fn} not supported")

        self.dense_in = nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())
        self.dense_out = nn.Dense(self.out_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())

        for i in range(self.num_blocks):

            block = [
                nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())
            ]

            blocks.append(block)

        self.blocks = blocks

    def __call__(self, t, theta, x=None, train=True, context=None):

        if not isinstance(t, jnp.ndarray):
            t = jnp.array([t], dtype=theta.dtype)
        elif isinstance(t, jnp.ndarray) and len(t.shape) == 0:
            t = t.astype(dtype=theta.dtype)
            t = jnp.expand_dims(t, 0)

        if len(t.shape) == 1:
            t = jnp.expand_dims(t, 1)

        if x is None:
            x = jnp.zeros((theta.shape[0], 0))

        x = jnp.concatenate([x, t, theta], axis=-1)

        x = self.dense_in(x)

        for block in self.blocks:

            x_ = x

            for layer in block:
                x = layer(x)
                x = self.activation(x)

            x += x_

        x = self.dense_out(x)

        return x

class DenseResidualBlocks(nn.Module):

    hidden_dim: int
    activation_fn: str = 'elu'
    context_dim: int = 0
    use_layer_norm: bool = False 

    def setup(self) -> None:

        if self.activation_fn == 'elu':
            self.activation = nn.elu
        elif self.activation_fn == 'gelu':
            self.activation = nn.gelu
        else:
            raise ValueError(f"Activation function {self.activation_fn} not supported")

        self.layer1 = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform(),
                               bias_init=pytorch_bias_init(self.hidden_dim))
        self.layer2 = nn.Dense(self.hidden_dim, kernel_init=uniform_shifted(), bias_init=uniform_shifted())

        # Add layer normalization
        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm()
            self.norm2 = nn.LayerNorm()

        if self.context_dim > 0:
            self.context_layer = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform(),
                                          bias_init=pytorch_bias_init(self.context_dim))

    def __call__(self, x, context = None):


        x_skip = x
        x = self.activation(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        if context is not None:
            x = nn.glu(jnp.concatenate(
                [x, self.context_layer(context)], axis=1), axis=1)

        return x + x_skip

class Identity(nn.Module):

        def __call__(self, x):
            return x

class SBIResidualNet(nn.Module):

    hidden_dims: List[int]
    out_dim: int
    in_dim: int
    activation_fn: str = 'elu'

    def setup(self):

        self.model = DenseResidualNet(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            in_dim=self.in_dim,
            context_dim=0,
            activation_fn=self.activation_fn
        )

    def __call__(self, t, theta, x, train=True):

        if not isinstance(t, jnp.ndarray):
            t = jnp.array([t], dtype=x.dtype)
        elif isinstance(t, jnp.ndarray) and len(t.shape) == 0:
            t = t.astype(dtype=x.dtype)
            t = jnp.expand_dims(t, 0)

        if len(t.shape) == 1:
            t = jnp.expand_dims(t, 1)

        x = jnp.concatenate([x, t, theta], axis=-1)

        x = self.model(x, context=None, train=train)

        return x

class DenseResidualNet(nn.Module):

    hidden_dims: List[int]
    out_dim: int
    in_dim: int
    context_dim: int = 0
    activation_fn: str = 'elu'
    use_layer_norm: bool = False
    use_batch_norm: bool = False

    def setup(self):

        blocks = []
        projections = []

        self.dense_in = nn.Dense(self.hidden_dims[0], kernel_init=lecun_uniform(),
                                 bias_init=pytorch_bias_init(self.in_dim))
         # Add input normalization
        
        #Add batch norm
        if self.use_batch_norm:
            self.loss_grad_bn = nn.BatchNorm()

        for hidden_dim, hidden_dim_next in zip(self.hidden_dims, list(self.hidden_dims[1:]) + [self.out_dim]):
            blocks.append(DenseResidualBlocks(hidden_dim, context_dim = self.context_dim,
                                              activation_fn=self.activation_fn,
                                              use_layer_norm=False))
            if hidden_dim != hidden_dim_next:
                projections.append(nn.Dense(hidden_dim_next, use_bias=True,
                                           kernel_init=lecun_uniform(),
                                           bias_init=pytorch_bias_init(hidden_dim)))
            else:
                projections.append(Identity())

        self.blocks = blocks
        self.projections = projections

    def __call__(self, theta, context = None, train=True):


        if self.use_batch_norm:
            theta = self.loss_grad_bn(theta, use_running_average=not train)
    
        x = self.dense_in(theta)

        for block, projection in zip(self.blocks, self.projections):

            x = block(x, context=context)
            x = projection(x)

        return x
    
class DenseResidualNet_batchnorm(nn.Module):

    hidden_dims: List[int]
    out_dim: int
    in_dim: int
    context_dim: int = 0
    activation_fn: str = 'elu'
    use_layer_norm: bool = False


    def setup(self):

        blocks = []
        projections = []

        self.dense_in = nn.Dense(self.hidden_dims[0], kernel_init=lecun_uniform(),
                                 bias_init=pytorch_bias_init(self.in_dim))
         # Add input normalization
        
        #Add batch norm
        self.loss_grad_bn = nn.BatchNorm()

        for hidden_dim, hidden_dim_next in zip(self.hidden_dims, list(self.hidden_dims[1:]) + [self.out_dim]):
            blocks.append(DenseResidualBlocks(hidden_dim, context_dim = self.context_dim,
                                              activation_fn=self.activation_fn,
                                              use_layer_norm=False))
            if hidden_dim != hidden_dim_next:
                projections.append(nn.Dense(hidden_dim_next, use_bias=True,
                                           kernel_init=lecun_uniform(),
                                           bias_init=pytorch_bias_init(hidden_dim)))
            else:
                projections.append(Identity())

        self.blocks = blocks
        self.projections = projections

    def __call__(self, theta, context = None, train=True):

        with jax.checking_leaks():
            x = self.loss_grad_bn(theta, use_running_average=not train)
    
        x = self.dense_in(x)

        for block, projection in zip(self.blocks, self.projections):

            x = block(x, context=context)
            x = projection(x)

        return x

class ShortcutMLP(nn.Module):

    hidden_dims: List[int]
    out_dim: int
    in_dim: int
    activation_fn: str = 'elu'

    def setup(self):

        self.model = DenseResidualNet(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            in_dim=self.in_dim,
            context_dim=0,
            activation_fn=self.activation_fn
        )

    def __call__(self, x, y, t, d, train=True):

        if not isinstance(t, jnp.ndarray):
            t = jnp.array([t], dtype=x.dtype)
        elif isinstance(t, jnp.ndarray) and len(t.shape) == 0:
            t = t.astype(dtype=x.dtype)
            t = jnp.expand_dims(t, 0)
        if len(t.shape) == 1:
            t = jnp.expand_dims(t, 1)

        if not isinstance(d, jnp.ndarray):
            d = jnp.array([d], dtype=x.dtype)
        elif isinstance(d, jnp.ndarray) and len(d.shape) == 0:
            d = d.astype(dtype=x.dtype)
            d = jnp.expand_dims(d, 0)
        if len(d.shape) == 1:
            d = jnp.expand_dims(d, 1)

        x = jnp.concatenate([x, t, d, y], axis=-1)

        x = self.model(x, context=None, train=train)

        return x

class BenchmarkFMPE(ContinuousNormalizingFlow):

    hidden_dim: int = 64
    num_blocks: int = 10
    out_dim: int = 2

    def setup(self):
        super().setup()

        self.model = BaseBenchmarkFMPE(
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            out_dim=self.out_dim
        )



# --- Helper Modules ---

class FlaxTimesteps(nn.Module):
    """
    A Flax module for creating sinusoidal timestep embeddings.
    This is a standard component in diffusion and flow-matching models.
    
    Attributes:
        num_channels: The number of channels in the output embedding.
        flip_sin_to_cos: Whether to flip the order of sin and cos in the embedding.
        downscale_freq_shift: A frequency shift parameter.
    """
    num_channels: int
    flip_sin_to_cos: bool = True
    downscale_freq_shift: float = 0.0

    @nn.compact
    def __call__(self, timesteps: Array) -> Array:
        """
        Args:
            timesteps: A 1D array of timesteps.
        Returns:
            An array of shape (timesteps.shape[0], num_channels)
        """
        half_dim = self.num_channels // 2
        exponent = -jnp.log(10000.0) * jnp.arange(start=0, stop=half_dim, dtype=jnp.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        
        emb = jnp.exp(exponent)
        emb = timesteps[:, None] * emb[None, :]
        
        if self.flip_sin_to_cos:
            emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)
        else:
            emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class Downsample(nn.Module):
    """A downsampling block using a strided convolution."""
    features: int
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return nn.Conv(self.features, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)

class Upsample(nn.Module):
    """An upsampling block using a transposed convolution."""
    features: int
    @nn.compact
    def __call__(self, x: Array) -> Array:
        return nn.ConvTranspose(self.features, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)

class ConvResidualBlock(nn.Module):
    """
    A convolutional residual block with conditioning.

    Attributes:
        features: The number of output channels.
        activation_fn: The activation function to use.
        norm_groups: The number of groups for Group Normalization.
    """
    features: int
    activation_fn: Any = nn.silu
    norm_groups: int = 8
    
    @nn.compact
    def __call__(self, x: Array, cond_emb: Array, train: bool = True) -> Array:
        """
        Args:
            x: Input feature map of shape (B, H, W, C).
            cond_emb: Conditioning embedding of shape (B, D).
            train: A boolean flag (not used in this block but good practice).
        
        Returns:
            Output feature map of shape (B, H, W, features).
        """
        h = nn.GroupNorm(num_groups=self.norm_groups)(x)
        h = self.activation_fn(h)
        h = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(h)

        # Project conditioning embedding and add to the feature map
        cond_proj = nn.Dense(self.features)(self.activation_fn(cond_emb))
        h = h + cond_proj[:, None, None, :] # Broadcast to spatial dimensions

        h = nn.GroupNorm(num_groups=self.norm_groups)(h)
        h = self.activation_fn(h)
        h = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(h)

        # Residual connection
        if x.shape[-1] != self.features:
            x_proj = nn.Conv(self.features, kernel_size=(1, 1))(x)
            return x_proj + h
        return x + h

# --- Main Embedding Network ---

class ConvolutionalEmbeddingNet(nn.Module):
    """
    A U-Net based Convolutional Embedding Network for Flow Matching.

    This network takes simulation data `x`, parameters `theta`, and a time `t` as
    input, and outputs a vector field of the same spatial dimensions as `x`.
    
    Attributes:
        out_channels: Number of channels in the output.
        start_channels: Number of channels in the first convolutional layer.
        dim_mults: A tuple of channel multipliers for each U-Net level.
        num_residual_blocks: Number of residual blocks per level.
        time_emb_dim: The dimension for the time embedding.
        theta_emb_dim: The dimension for the parameter embedding. Set to 0 to disable.
        activation_fn: The activation function for the network.
        norm_groups: The number of groups for Group Normalization.
    """
    out_channels: int
    start_channels: int = 64
    dim_mults: Tuple[int, ...] = (1, 2, 4)
    num_residual_blocks: int = 2
    time_emb_dim: int = 32
    theta_emb_dim: int = 32
    activation_fn: Any = nn.silu
    norm_groups: int = 8

    @nn.compact
    def __call__(self, t: Array, theta: Array, x: Array, train: bool = True) -> Array:
        """
        The forward pass of the network.

        Args:
            t: A batch of time values, shape (B,).
            theta: A batch of simulation parameters, shape (B, param_dim).
            x: A batch of simulation data, shape (B, H, W, C).
            train: A boolean indicating if the model is in training mode.
        
        Returns:
            The output vector field from the network, shape (B, H, W, out_channels).
        """
        # --- 1. Input Handling & Conditioning Embeddings ---
        if not isinstance(t, jnp.ndarray) or t.ndim == 0:
            t = jnp.atleast_1d(t).astype(x.dtype)

        # Time embedding
        time_emb_module = FlaxTimesteps(num_channels=self.time_emb_dim)
        t_emb = time_emb_module(t)
        
        time_mlp_dim = self.time_emb_dim * 4
        t_emb = nn.Sequential([
            nn.Dense(time_mlp_dim),
            self.activation_fn,
            nn.Dense(time_mlp_dim),
        ], name="time_mlp")(t_emb)
        
        # Combine time and theta embeddings
        cond_emb = t_emb
        if self.theta_emb_dim > 0 and theta is not None:
            theta_mlp_dim = self.theta_emb_dim * 4
            theta_emb = nn.Sequential([
                nn.Dense(theta_mlp_dim),
                self.activation_fn,
                nn.Dense(time_mlp_dim) # Project to same dim as time_mlp output
            ], name="theta_mlp")(theta)
            cond_emb += theta_emb

        # --- 2. U-Net Architecture ---
        
        # -- Initial Convolution --
        h = nn.Conv(self.start_channels, kernel_size=(3, 3), padding="SAME")(x)
        skips = [h]

        # -- Downsampling Path --
        current_ch = self.start_channels
        for i, mult in enumerate(self.dim_mults):
            out_ch = self.start_channels * mult
            for _ in range(self.num_residual_blocks):
                h = ConvResidualBlock(
                    features=out_ch,
                    activation_fn=self.activation_fn,
                    norm_groups=self.norm_groups,
                )(h, cond_emb, train=train)
                skips.append(h)
            
            is_last = (i == len(self.dim_mults) - 1)
            if not is_last:
                h = Downsample(features=out_ch)(h)
                skips.append(h)
            current_ch = out_ch

        # -- Bottleneck --
        h = ConvResidualBlock(current_ch, activation_fn=self.activation_fn, norm_groups=self.norm_groups)(h, cond_emb, train)
        h = ConvResidualBlock(current_ch, activation_fn=self.activation_fn, norm_groups=self.norm_groups)(h, cond_emb, train)

        # -- Upsampling Path --
        for i, mult in enumerate(reversed(self.dim_mults)):
            out_ch = self.start_channels * mult
            is_last = (i == len(self.dim_mults) - 1)
            if not is_last:
                h = jnp.concatenate([h, skips.pop()], axis=-1)
                h = Upsample(features=out_ch)(h)
                
            for _ in range(self.num_residual_blocks + 1):
                h = jnp.concatenate([h, skips.pop()], axis=-1)
                h = ConvResidualBlock(
                    features=out_ch,
                    activation_fn=self.activation_fn,
                    norm_groups=self.norm_groups,
                )(h, cond_emb, train=train)

        # -- Final Projection --
        final_conv = nn.Sequential([
            nn.GroupNorm(num_groups=self.norm_groups),
            self.activation_fn,
            nn.Conv(
                features=self.out_channels,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_init=nn.initializers.zeros,
            ),
        ])
        
        return final_conv(h)
    

"""
Updated Theta-Context Transformer

Changes:
- `theta` is (B, dim_flow) and each scalar along dim_flow is treated as a token.
- `t` is embedded as a separate token and appended to Q and K.
- `context` is now of shape (B, dim_flow+1). Each scalar along that dimension is treated as a token (so context aligns with theta+t). LayerNorm is applied per-token before embedding.
- All inputs become sequences of tokens that are embedded into embed_dim vectors.
- Attention is done with configurable heads. Context tokens are included in K (always) and optionally in V.
- Feed-forward network is applied token-wise to theta tokens.
"""

from typing import Optional, Callable

import jax.numpy as jnp
import flax.linen as nn


class ThetaContextTransformer(nn.Module):
    embed_dim: int
    num_heads: int = 8
    include_context_in_v: bool = True
    dropout_rate: float = 0.0
    use_bias: bool = True
    feed_forward_ctor: Optional[Callable[[], nn.Module]] = None

    def setup(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = self.embed_dim // self.num_heads

        # Embeddings
        self.theta_embed = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.t_embed = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.context_norm = nn.LayerNorm()
        self.context_embed = nn.Dense(self.embed_dim, use_bias=self.use_bias)

        # Q/K/V projections
        self.q_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.k_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.v_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)

        self.out_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)

        if self.feed_forward_ctor is not None:
            self.feed_forward = self.feed_forward_ctor()
        else:
            self.feed_forward = None

        self.attn_dropout = nn.Dropout(rate=self.dropout_rate)
        self.out_dropout = nn.Dropout(rate=self.dropout_rate)
        # Add output projection to map back to parameter space
        self.output_proj = nn.Dense(1, use_bias=self.use_bias)  # Project to scalar per token

    def _split_heads(self, x):
        B, L, E = x.shape
        x = x.reshape(B, L, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x):
        B, H, L, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, L, H * D)

    @nn.compact
    def __call__(self, theta: jnp.ndarray, context: Optional[jnp.ndarray] = None, t: Optional[jnp.ndarray] = None, *, deterministic: bool = True):
        """
        Parameters
        ----------
        theta: (B, L)  -- dim_flow scalars, each a token
        context: (B, L+1)  -- context correction terms, each a token
        t: (B,) or (B,1) -- scalar per batch, becomes its own token

        Returns
        -------
        theta_updated: (B, L, embed_dim) -- updated token embeddings for theta tokens
        info: dict with 'attn_weights'
        """
        B, L = theta.shape

        # --- embed theta tokens ---
        theta_in = theta[..., None]  # (B, L, 1)
        theta_e = self.theta_embed(theta_in)  # (B, L, E)

        # --- embed t ---
        if t is not None:
            t_arr = jnp.asarray(t)
            if t_arr.ndim == 1:
                t_arr = t_arr[:, None]
            t_e = self.t_embed(t_arr)[:, None, :]  # (B, 1, E)
        else:
            t_e = None

        # --- embed context tokens ---
        if context is not None:
            # if context.shape[1] != L :
            #     raise ValueError("context must have shape (B, L)")
            # treat each scalar as a token
            context_in = context[..., None]  # (B, L+1, 1)
            # apply norm per token
            context_normed = self.context_norm(context_in)
            ctx_e = self.context_embed(context_normed)  # (B, L+1, E)
        else:
            ctx_e = None

        # --- Build Q ---
        q_parts = [theta_e]
        if t_e is not None:
            q_parts.append(t_e)
        q_seq = jnp.concatenate(q_parts, axis=1)  # (B, Lq, E)

        # --- Build K ---
        k_parts = [theta_e]
        if t_e is not None:
            k_parts.append(t_e)
        if ctx_e is not None:
            k_parts.append(ctx_e)
        k_seq = jnp.concatenate(k_parts, axis=1)

        # --- Build V ---
        v_parts = [theta_e]
        if t_e is not None:
            v_parts.append(t_e)
        if self.include_context_in_v and ctx_e is not None:
            v_parts.append(ctx_e)
        v_seq = jnp.concatenate(v_parts, axis=1)

        # --- Project Q, K, V ---
        Q = self.q_proj(q_seq)
        K = self.k_proj(k_seq)
        V = self.v_proj(v_seq)

        # --- Attention ---
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_logits = jnp.einsum('bhqd,bhkd->bhqk', Qh, Kh) * scale
        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, deterministic=deterministic)

        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, Vh)
        attn_out_comb = self._combine_heads(attn_out)
        attn_out_comb = self.out_proj(attn_out_comb)
        attn_out_comb = self.out_dropout(attn_out_comb, deterministic=deterministic)

        # take updated first L positions (theta tokens)
        theta_attn_updated = attn_out_comb[:, :L, :] + theta_e

        # --- Feed-forward per theta token ---
        if self.feed_forward is not None:
            tokens_flat = theta_attn_updated.reshape(B * L, self.embed_dim)
            if ctx_e is not None:
                ctx_flat = ctx_e.reshape(B, -1)
                ctx_tiled = jnp.repeat(ctx_flat[:, None, :], L, axis=1).reshape(B * L, -1)
                ff_input = jnp.concatenate([tokens_flat, ctx_tiled], axis=-1)
            else:
                ff_input = tokens_flat
            ff_out = self.feed_forward(ff_input)
            if ff_out.shape[-1] != self.embed_dim:
                raise ValueError("feed_forward must output last dim == embed_dim")
            theta_ff = ff_out.reshape(B, L, self.embed_dim)
            theta_updated = theta_attn_updated + theta_ff
        else:
            theta_updated = theta_attn_updated

        # Project back to original parameter space
        theta_correction = self.output_proj(theta_updated)  # (B, L, 1)
        theta_correction = theta_correction.squeeze(-1)     # (B, L)
        
        return theta_correction  # Same shape as input theta


class ThetaContextTransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int = 8
    include_context_in_v: bool = True
    dropout_rate: float = 0.0
    use_bias: bool = True
    feed_forward_ctor: Optional[Callable[[], nn.Module]] = None

    def setup(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = self.embed_dim // self.num_heads

        # Embeddings
        self.theta_embed = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.t_embed = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.context_norm = nn.LayerNorm()
        self.context_embed = nn.Dense(self.embed_dim, use_bias=self.use_bias)

        # Q/K/V projections
        self.q_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.k_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.v_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)

        self.out_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)

        # Residual LayerNorms
        self.attn_norm = nn.LayerNorm()
        self.ffn_norm = nn.LayerNorm()

        # Feed-forward network
        if self.feed_forward_ctor is not None:
            self.feed_forward = self.feed_forward_ctor()
        else:
            self.feed_forward = nn.Sequential([
                nn.Dense(self.embed_dim * 4, use_bias=self.use_bias),
                nn.gelu,
                nn.Dense(self.embed_dim, use_bias=self.use_bias),
            ])

        self.attn_dropout = nn.Dropout(rate=self.dropout_rate)
        self.out_dropout = nn.Dropout(rate=self.dropout_rate)

        # Project back to parameter space (scalar per token)
        self.output_proj = nn.Dense(1, use_bias=self.use_bias)

    def _split_heads(self, x):
        B, L, E = x.shape
        x = x.reshape(B, L, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x):
        B, H, L, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, L, H * D)

    def __call__(self, theta: jnp.ndarray, context: Optional[jnp.ndarray] = None,
                 t: Optional[jnp.ndarray] = None, *, deterministic: bool = True):
        """
        Parameters
        ----------
        theta: (B, L)        -- dim_flow scalars, each a token
        context: (B, L+1)    -- context correction terms, each a token
        t: (B,) or (B,1)     -- scalar per batch, becomes its own token

        Returns
        -------
        theta_correction: (B, L) -- updated corrections for theta
        """
        B, L = theta.shape

        # --- Embed theta ---
        theta_in = theta[..., None]  # (B, L, 1)
        theta_e = self.theta_embed(theta_in)  # (B, L, E)

        # --- Embed t ---
        if t is not None:
            t_arr = jnp.asarray(t)
            if t_arr.ndim == 1:
                t_arr = t_arr[:, None]
            t_e = self.t_embed(t_arr)[:, None, :]  # (B, 1, E)
        else:
            t_e = None

        # --- Embed context ---
        if context is not None:
            context_in = context[..., None]  # (B, L+1, 1)
            context_normed = self.context_norm(context_in)
            ctx_e = self.context_embed(context_normed)  # (B, L+1, E)
        else:
            ctx_e = None

        # =====================
        # Multi-head Attention (with residual + pre-norm)
        # =====================
        attn_in = self.attn_norm(theta_e)

        # Q sequence: theta (+ t)
        q_parts = [attn_in]
        if t_e is not None:
            q_parts.append(t_e)
        q_seq = jnp.concatenate(q_parts, axis=1)  # (B, Lq, E)

        # K/V sequence: theta (+ t) (+ context)
        kv_parts = [attn_in]
        if t_e is not None:
            kv_parts.append(t_e)
        if ctx_e is not None:
            kv_parts.append(ctx_e)
        kv_seq = jnp.concatenate(kv_parts, axis=1)  # (B, Lkv, E)

        # Project Q/K/V
        Q = self.q_proj(q_seq)
        K = self.k_proj(kv_seq)
        if self.include_context_in_v:
            V = self.v_proj(kv_seq)
        else:
            V = self.v_proj(attn_in)  # same length as K

        # Attention computation
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_logits = jnp.einsum('bhqd,bhkd->bhqk', Qh, Kh) * scale
        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, deterministic=deterministic)

        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, Vh)
        attn_out_comb = self._combine_heads(attn_out)
        attn_out_comb = self.out_proj(attn_out_comb)
        attn_out_comb = self.out_dropout(attn_out_comb, deterministic=deterministic)

        attn_out_comb = attn_out_comb[:, :L, :]  # ensure shape (B, L, E)
        x = theta_e + attn_out_comb
        # Residual connection
        x = theta_e + attn_out_comb

        # =====================
        # Feed-forward (with residual + pre-norm)
        # =====================
        ffn_in = self.ffn_norm(x)
        ffn_out = self.feed_forward(ffn_in)
        x = x + ffn_out  # residual

        # Project back to parameter space
        theta_correction = self.output_proj(x)  # (B, L, 1)
        theta_correction = theta_correction.squeeze(-1)  # (B, L)

        return theta_correction




class ThetaContextTransformerStack(nn.Module):
    embed_dim: int
    num_heads: int = 8
    num_layers: int = 4
    include_context_in_v: bool = True
    dropout_rate: float = 0.0
    use_bias: bool = True
    feed_forward_ctor: Optional[Callable[[], nn.Module]] = None

    @nn.compact
    def __call__(self, theta, context=None, t=None, deterministic=True):
        """
        theta:   (B, L)
        context: (B, L+1)
        t:       (B,) or (B,1)
        """
        x = theta
        for i in range(self.num_layers):
            x = ThetaContextTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                include_context_in_v=self.include_context_in_v,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                feed_forward_ctor=self.feed_forward_ctor,
                name=f"block_{i}"
            )(x, context=context, t=t, deterministic=deterministic)
        return x