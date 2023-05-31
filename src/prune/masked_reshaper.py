import jax
import jax.numpy as jnp
from typing import Optional
from functools import partial
from flax.traverse_util import flatten_dict, unflatten_dict


def apply_mask(params, masks):
    return jax.tree_util.tree_map(lambda x, y: x * y, params, masks)


def get_mask_ids(masks):
    m_id, bool_idxs = [0], []
    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(masks).items()
    }

    for m in flat_params:
        bool_idx = flat_params[m] > 0
        shape_flat = jnp.sum(bool_idx)
        m_id.append(int(m_id[-1] + shape_flat))
        bool_idxs.append(bool_idx)
    return m_id, bool_idxs


class MaskedParameterReshaper(object):
    def __init__(
        self, masks, n_devices: Optional[int] = None, verbose: bool = False
    ):
        self.masks = masks
        self.flat_masks = {
            "/".join(k): jnp.array(v).astype(jnp.float32)
            for k, v in flatten_dict(masks).items()
        }

        self.flat_sigmas = {
            "/".join(k): 1e20 * (1 - jnp.array(v).astype(jnp.float32))
            for k, v in flatten_dict(masks).items()
        }
        self.unflat_shape = jax.tree_map(jnp.shape, masks)
        self.network_shape = jax.tree_map(jnp.shape, self.flat_masks)
        self.mask_id, self.bool_idxs = get_mask_ids(masks)
        self.params_to_opt = sum(
            jnp.sum(x != 0.0) for x in jax.tree_leaves(masks)
        )

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
        if self.n_devices > 1 and verbose:
            print(
                "ParameterReshaper: More than one device detected. Please make"
                " sure that the ES population size divides evenly across the"
                " number of devices to pmap/parallelize over."
            )

    @partial(jax.jit, static_argnums=0)
    def to_mask(self, x):
        """Perform reshaping for a 2D matrix (pop_members, params)."""
        vmap_shape = jax.vmap(self.flat_to_mask, in_axes=(0,))
        if self.n_devices > 1:
            x = self.split_params_for_pmap(x)
            map_shape = jax.pmap(vmap_shape)
        else:
            map_shape = vmap_shape
        return map_shape(x)

    @partial(jax.jit, static_argnums=0)
    def to_flat(self, masked_params):
        return jax.vmap(self.mask_to_flat, in_axes=(0,))(masked_params)

    def mask_to_flat(self, params):
        flat = []
        flat_masked = {
            "/".join(k): v.astype(jnp.float32)
            for k, v in flatten_dict(params).items()
        }
        layer_keys = self.network_shape.keys()
        for i, p_k in enumerate(layer_keys):
            flat.append(
                flat_masked[p_k].reshape(
                    -1,
                )
            )
        # Return only non-zero elements
        return jnp.concatenate(flat)

    def split_params_for_pmap(self, x):
        """Helper reshapes param (bs, #params) into (#dev, bs/#dev, #params)."""
        return jnp.stack(jnp.split(x, self.n_devices))

    @partial(jax.jit, static_argnums=0)
    def flat_to_mask(self, flat_params):
        masked_params = {}
        layer_keys = self.network_shape.keys()

        # Loop over layers in network
        for i, p_k in enumerate(layer_keys):
            # Select params from flat to vector to be reshaped
            p_flat = jax.lax.dynamic_slice(
                flat_params,
                (self.mask_id[i],),
                (self.mask_id[i + 1] - self.mask_id[i],),
            )

            new_params = self.flat_masks[p_k].at[self.bool_idxs[i]].set(p_flat)
            masked_params[p_k] = new_params
        return unflatten_dict(
            {tuple(k.split("/")): v for k, v in masked_params.items()}
        )

    @partial(jax.jit, static_argnums=0)
    def flat_to_sigma(self, flat_params):
        masked_params = {}
        layer_keys = self.network_shape.keys()

        # Loop over layers in network
        for i, p_k in enumerate(layer_keys):
            # Select params from flat to vector to be reshaped
            p_flat = jax.lax.dynamic_slice(
                flat_params,
                (self.mask_id[i],),
                (self.mask_id[i + 1] - self.mask_id[i],),
            )

            new_params = self.flat_sigmas[p_k].at[self.bool_idxs[i]].set(p_flat)
            masked_params[p_k] = new_params
        return unflatten_dict(
            {tuple(k.split("/")): v for k, v in masked_params.items()}
        )

    def masked_to_flat_sparse(self, masked_params):
        flat = []
        flat_masked = {
            "/".join(k): v.astype(jnp.float32)
            for k, v in flatten_dict(masked_params).items()
        }
        layer_keys = self.network_shape.keys()
        for i, p_k in enumerate(layer_keys):
            non_zero = flat_masked[p_k][flat_masked[p_k] != 0.0].reshape(
                -1,
            )
            flat.append(non_zero)
        # Return only non-zero elements
        return jnp.concatenate(flat)

    @property
    def vmap_dict(self):
        """Get a dictionary specifying axes to vmap over."""
        vmap_dict = jax.tree_map(lambda x: 0, self.masks)
        return vmap_dict
