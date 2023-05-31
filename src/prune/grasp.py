import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from .snip import get_data_loaders


def get_grasp_scores(model, params, dataset_name: str):
    """Calculate gradient-weight product for each parameter and prune."""
    X, y = get_data_loaders(dataset_name, 1000)

    @jax.jit
    def get_grads_hessian(params, images, labels):
        """Compute grads, loss and accuracy (single batch)."""

        def loss_fn(params):
            logits = model.apply(params, images)
            one_hot = jax.nn.one_hot(labels, 10)
            cent_loss = optax.softmax_cross_entropy(
                logits=logits, labels=one_hot
            )
            loss = jnp.mean(cent_loss)
            return loss

        grads = jax.grad(loss_fn)(params)
        hessian = jax.hessian(loss_fn)(params)
        return grads, hessian

    grads, hessian = get_grads_hessian(params, X, y)
    # Filter only prunable parameters with hessian respect to self
    # Compute -w * hessian * grad as scores
    # Remove weights with highest scores that reduce grad flow most

    scores = {}
    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(params)).items()
    }
    flat_grads = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(grads)).items()
    }
    flat_hessians = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(hessian)).items()
    }

    for m in flat_params:
        m_key = tuple(m.split("/"))
        h_key = m + "/" + m
        scores[m_key] = (
            -1
            * flat_params[m]
            * (jnp.tensordot(flat_hessians[h_key], flat_grads[m], 2))
        )
    return FrozenDict(unflatten_dict(scores))


def global_grasp_prune_threshold(
    grasp_scores, sparsity_level, modules_not_to_prune=None
):
    all_weights = []
    if modules_not_to_prune is None:
        modules_not_to_prune = []

    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(grasp_scores)).items()
    }

    for m in flat_params:
        # Only consider specific modules and not their biases
        if m not in modules_not_to_prune and not m.endswith("bias"):
            all_weights.append(flat_params[m].reshape(-1))
    all_scores = jnp.concatenate(all_weights)
    # Assumes sparsity_level in [0, 1]!
    return jnp.percentile(all_scores, (1 - sparsity_level) * 100)


def global_grasp_mask_from_threshold(
    grasp_scores, threshold, modules_not_to_prune=None
):
    masks = {}
    if modules_not_to_prune is None:
        modules_not_to_prune = []

    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(grasp_scores)).items()
    }

    for m in flat_params:
        m_key = tuple(m.split("/"))
        if m not in modules_not_to_prune and not m.endswith("bias"):
            masks[m_key] = (flat_params[m] > threshold).astype(jnp.uint8)
        else:
            masks[m_key] = jnp.ones(flat_params[m].shape).astype(jnp.uint8)
    return FrozenDict(unflatten_dict(masks))
