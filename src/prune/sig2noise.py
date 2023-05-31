from flax.core import FrozenDict, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import jax.numpy as jnp


def global_sig2noise_prune_threshold(
    params, std, sparsity_level, modules_not_to_prune=None
):
    all_weights = []
    if modules_not_to_prune is None:
        modules_not_to_prune = []

    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(params)).items()
    }

    flat_std = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(std)).items()
    }

    for m in flat_params:
        # Only consider specific modules and not their biases
        if m not in modules_not_to_prune and not m.endswith("bias"):
            all_weights.append(
                jnp.abs(flat_params[m] / flat_std[m]).reshape(
                    -1,
                )
            )
    all_sig2noise = jnp.concatenate(all_weights)
    # Assumes sparsity_level in [0, 1]!
    return jnp.percentile(all_sig2noise, sparsity_level * 100)


def global_sig2noise_from_threshold(
    params, std, threshold, modules_not_to_prune=None, prune_smaller=True
):
    masks = {}
    if modules_not_to_prune is None:
        modules_not_to_prune = []

    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(params)).items()
    }

    flat_std = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(std)).items()
    }

    for m in flat_params:
        m_key = tuple(m.split("/"))
        if m not in modules_not_to_prune and not m.endswith("bias"):
            if prune_smaller:
                masks[m_key] = (
                    jnp.abs(flat_params[m] / flat_std[m]) >= threshold
                ).astype(jnp.uint8)
            else:
                masks[m_key] = (
                    jnp.abs(flat_params[m] / flat_std[m]) <= threshold
                ).astype(jnp.uint8)
        else:
            masks[m_key] = jnp.ones(flat_params[m].shape).astype(jnp.uint8)
    return FrozenDict(unflatten_dict(masks))
