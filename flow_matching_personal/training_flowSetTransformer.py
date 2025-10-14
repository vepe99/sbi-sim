from autocvd import autocvd
autocvd(num_gpus = 1, interval=1)

from functools import partial

# timing and progress bars
from timeit import default_timer as timer
from tqdm import tqdm

# numerics
import jax
import jax.numpy as jnp
from jax import jit
import jax.random as jr

import optimistix as optx
import equinox as eqx
import optax
from flax.training.checkpoints import save_checkpoint


from flow_SetTransformer import Flow
from dataloader import OdisseoOTAllParametersPosition_newprior
from sampling_time import sample_time


dataloader_train = OdisseoOTAllParametersPosition_newprior(
            DATA_ROOT = '/export/data/vgiusepp/odisseo_data/data_varying_position_newprior/sbi_sim/data/sbi-benchmarks/odisseo_AllParametersPosition_newprior',
            # replacement = True,
            num_samples = 100_000,
            split= [0, 94999],
            seed= 0,
            num_steps = 2000,
            batch_size = 64,
            use_jax = False,
            use_log = True,)

dataloader_val = OdisseoOTAllParametersPosition_newprior(
            DATA_ROOT = '/export/data/vgiusepp/odisseo_data/data_varying_position_newprior/sbi_sim/data/sbi-benchmarks/odisseo_AllParametersPosition_newprior',
            num_samples= 100_000,
            split= [95000, 99999],
            seed= 0,
            num_steps= 1000,
            batch_size= 64,
            use_jax = False, 
            use_log=True)



# ============================
# Initialization
# ============================

SEED = 42
dim_flow = 13
N_dim = 64
N_head = 8
depth = 4
N_seed = 1
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
flow = Flow(dim_flow=dim_flow, N_dim=N_dim, N_head=N_head, depth=depth, N_seed=N_seed, sample_size=1000)

flow_params = flow.init_weights(subkey)


# ============================
# Loss function
# ============================

@jit
def loss_fn(flow_params, rng, batch):
    # v_pred = flow.apply({'params': flow_params}, timesteps=t, sample=theta, encoder_hidden_states=x_obs)
    # loss = jnp.mean((v_pred - v_target) ** 2)
    # return loss

    sigma_min = 0.0001

    time_alpha = 0.0

    # sample time
    t, rng = sample_time(rng, batch["parameters"].shape[0],
                            t0=0.0, t1=1.0,
                            alpha=time_alpha, eps=0.0)
    
    # sample noise
    epsilon = jr.normal(rng, shape=batch["parameters"].shape)
    noise = jnp.einsum("ab,a->ab", epsilon, (1 - (1-sigma_min) * t))

    # get x_t
    mu_x = jnp.einsum('a, ab->ab', t, batch["parameters"])
    theta_t = mu_x + noise

    # prediction of flow
    flow_prediction = flow.apply({'params': flow_params},
                                    timesteps=jnp.expand_dims(t, axis=1), 
                                    sample=theta_t,
                                    encoder_hidden_states=batch["conditioning"],)
    
    target = batch['parameters'] - (1-sigma_min) * epsilon

    loss = jnp.mean((target - flow_prediction) ** 2, axis=1)

    return jnp.mean(loss)



# ============================
# Optimizer
# ============================

learning_rate = 1e-3
optimizer = optax.adamw(learning_rate)
opt_state = optimizer.init(flow_params)

# ============================
# Training step
# ============================

@eqx.filter_jit
def train_step(flow_params, opt_state, rng, batch ):
    """
    Performs one optimization step.
    """
    
    loss_value, grads = jax.value_and_grad(loss_fn)(
        flow_params, rng, batch
    )

    updates, opt_state = optimizer.update(grads, opt_state, flow_params)
    flow_params = optax.apply_updates(flow_params, updates)
    return flow_params, opt_state, loss_value

# ============================
# Training loop
# ============================

print("Starting training with optax...")
training_loss = []
val_loss = []
n_epoch = 200

trained_params = flow_params
best_loss = float('inf')
best_params = trained_params

start_time = timer()
pbar = tqdm(range(n_epoch))


for epoch in pbar:
    epoch_training_loss = 0.0
    for batch_train_idx, batch in enumerate(dataloader_train):

        key, subkey = jax.random.split(key)
        trained_params, opt_state, loss_value = train_step(
            trained_params, opt_state,key, batch )
        
        epoch_training_loss += loss_value.item()

    epoch_val_loss = 0.0
    for batch_val_idx, batch in enumerate(dataloader_val):
        
        key, subkey = jax.random.split(key)
    
        val_loss_batch = loss_fn(trained_params, key, batch)
        
        epoch_val_loss += val_loss_batch.item()

    epoch_training_loss /= (batch_train_idx + 1)
    training_loss.append(epoch_training_loss)
    epoch_val_loss /= (batch_val_idx + 1)
    val_loss.append(epoch_val_loss)

    pbar.set_description(f"Epoch {epoch+1}, Loss: {epoch_training_loss:.6f}, Val Loss: {epoch_val_loss:.6f}", )

    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        best_params = trained_params
        print(f"New best model found at epoch {epoch+1} with val loss {best_loss:.6f}")

file_path = save_checkpoint('/export/home/vgiusepp/sbi_diff_sim/sbi-sim/flow_matching_personal/SetTransformer/', target=best_params, step=0, )
print(f"Model saved at {file_path}")