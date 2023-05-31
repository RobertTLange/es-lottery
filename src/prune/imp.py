import chex
from typing import List, Optional
from flax.core import FrozenDict, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
from src.prune.masked_reshaper import MaskedParameterReshaper, apply_mask
from src.prune.sig2noise import (
    global_sig2noise_prune_threshold,
    global_sig2noise_from_threshold,
)

from src.prune.snip import get_snip_grad_weights
from src.prune.grasp import (
    get_grasp_scores,
    global_grasp_prune_threshold,
    global_grasp_mask_from_threshold,
)


class IMPBaselines(object):
    def __init__(
        self,
        baseline_name: str,
        init_params: chex.ArrayTree,
        prune_ratio: float,
        modules_not_to_prune: List[str],
        model,
        fake_args: chex.Array,
        verbose: bool = True,
        dataset_name: Optional[str] = None,
    ):
        assert baseline_name in [
            "final-ticket",
            "final-ticket_permute",
            "final_permute-ticket_permute",
            "random-reinit",
            "sig2noise",
            "snip",
            "grasp",
            "mask-reinit",
        ]
        self.baseline_name = baseline_name
        # Initialization of model parameters
        self.init_params = init_params
        # Percentage of weights not pruned each iteration
        self.prune_ratio = prune_ratio
        # List of layers not pruned
        self.modules_not_to_prune = modules_not_to_prune
        # Model for reinitialization
        self.model = model
        # Fake arguments for reinitialization
        self.fake_args = fake_args
        self.verbose = verbose
        self.dataset_name = dataset_name

    def apply(
        self,
        rng: chex.PRNGKey,
        iter_id: int,
        params: chex.ArrayTree,
        std: Optional[chex.ArrayTree] = None,
    ):
        """Apply pruning to weights after training."""
        thresh = 0
        # Before start of training create a dense mask
        if iter_id == 0:
            sl, new_params, new_masks, thresh = self.final_ticket(
                iter_id, params
            )
        else:
            if self.baseline_name == "final-ticket":
                sl, new_params, new_masks, thresh = self.final_ticket(
                    iter_id, params
                )
            elif self.baseline_name == "final-ticket_permute":
                sl, new_params, new_masks = self.final_permute(
                    rng, iter_id, params
                )
            elif self.baseline_name == "final_permute-ticket_permute":
                sl, new_params, new_masks = self.permute_permute(
                    rng, iter_id, params
                )
            elif self.baseline_name == "random-reinit":
                sl, new_params, new_masks = self.random_reinit(
                    rng, iter_id, params
                )
            elif self.baseline_name == "movement":
                sl, new_params, new_masks = self.movement(iter_id, params)
            elif self.baseline_name == "sig2noise":
                sl, new_params, new_masks, thresh = self.sig2noise_ticket(
                    iter_id, params, std
                )
            elif self.baseline_name == "snip":
                sl, new_params, new_masks = self.snip(iter_id, params)
            elif self.baseline_name == "grasp":
                sl, new_params, new_masks = self.grasp(iter_id, params)
            elif self.baseline_name == "mask-reinit":
                sl, new_params, new_masks = self.mask_reinit(
                    rng, iter_id, params
                )
            else:
                raise ValueError("Invalid baseline method.")

        self.last_masks = new_masks
        summary = get_sparsity(new_masks, self.modules_not_to_prune)
        if self.verbose:
            print(iter_id, sl, summary)
        return sl, new_params, new_masks, summary, thresh

    def final_ticket(self, iter_id: int, params: chex.ArrayTree):
        """Mask: Final W Magnitudes + W: Original Init. - Winning Ticket"""
        sparsity_level = imp_iter_sparsity(iter_id, self.prune_ratio)
        thresh = global_magnitude_prune_threshold(
            params,
            sparsity_level,
            self.modules_not_to_prune,
        )
        new_masks = global_mask_from_threshold(
            params,
            thresh,
            self.modules_not_to_prune,
        )
        return sparsity_level, self.init_params, new_masks, thresh

    def sig2noise_ticket(
        self, iter_id: int, params: chex.ArrayTree, std: chex.ArrayTree
    ):
        """Mask: Final W Magnitudes + W: Original Init. - Winning Ticket"""
        sparsity_level = imp_iter_sparsity(iter_id, self.prune_ratio)
        thresh = global_sig2noise_prune_threshold(
            params,
            std,
            sparsity_level,
            self.modules_not_to_prune,
        )
        new_masks = global_sig2noise_from_threshold(
            params,
            std,
            thresh,
            self.modules_not_to_prune,
        )
        return sparsity_level, self.init_params, new_masks, thresh

    def final_permute(self, rng, iter_id, params):
        """Mask: Final W Magnitudes + W: Layerwise Permuted"""
        sparsity_level = imp_iter_sparsity(iter_id, self.prune_ratio)
        thresh = global_magnitude_prune_threshold(
            params,
            sparsity_level,
            self.modules_not_to_prune,
        )
        new_masks = global_mask_from_threshold(
            params,
            thresh,
            self.modules_not_to_prune,
        )
        masked_init = apply_mask(self.init_params, new_masks)
        permuted_params = get_final_permuted(
            rng, masked_init, self.modules_not_to_prune
        )
        return sparsity_level, permuted_params, new_masks

    def permute_permute(self, rng, iter_id, params):
        """Mask: Layerwise Permuted + W: Layerwise Permuted"""
        sparsity_level = imp_iter_sparsity(iter_id, self.prune_ratio)
        thresh = global_magnitude_prune_threshold(
            params,
            sparsity_level,
            self.modules_not_to_prune,
        )
        new_masks = global_mask_from_threshold(
            params,
            thresh,
            self.modules_not_to_prune,
        )
        masked_init = apply_mask(self.init_params, new_masks)
        permuted_params, permuted_masks = get_permuted_permuted(
            rng,
            masked_init,
            new_masks,
            self.modules_not_to_prune,
        )
        return sparsity_level, permuted_params, permuted_masks

    def random_reinit(self, rng, iter_id, params):
        """Mask: Random sparsity matched + W: Reinitialized"""
        sparsity_level = imp_iter_sparsity(iter_id, self.prune_ratio)
        thresh = global_magnitude_prune_threshold(
            params,
            sparsity_level,
            self.modules_not_to_prune,
        )
        new_masks = global_mask_from_threshold(
            params,
            thresh,
            self.modules_not_to_prune,
        )

        rng_reinit, rng_masks = jax.random.split(rng)
        try:
            reinit_params = self.model.init(rng_reinit, self.fake_args)
        except Exception:
            reinit_params = self.model.init(rng_reinit)
        masked_temp = apply_mask(reinit_params, new_masks)
        sampled_masks = get_random_masks(
            rng_masks, masked_temp, self.modules_not_to_prune
        )
        return sparsity_level, reinit_params, sampled_masks

    def mask_reinit(self, rng, iter_id, params):
        """Mask: IMP sparsity matched + W: Reinitialized"""
        sparsity_level = imp_iter_sparsity(iter_id, self.prune_ratio)
        thresh = global_magnitude_prune_threshold(
            params,
            sparsity_level,
            self.modules_not_to_prune,
        )
        new_masks = global_mask_from_threshold(
            params,
            thresh,
            self.modules_not_to_prune,
        )

        rng_reinit, rng_masks = jax.random.split(rng)
        try:
            reinit_params = self.model.init(rng_reinit, self.fake_args)
        except Exception:
            reinit_params = self.model.init(rng_reinit)
        return sparsity_level, reinit_params, new_masks

    def movement(self, iter_id, params):
        """Prune weights with smallest movement |w_f - w_i|"""
        sparsity_level = imp_iter_sparsity(iter_id, self.prune_ratio)
        # Construct movement matrix from init and final params
        masked_init = apply_mask(self.init_params, self.last_masks)
        movement_params = jax.tree_map(
            lambda x, y: jnp.abs(x - y), params, unfreeze(masked_init["params"])
        )
        thresh = global_magnitude_prune_threshold(
            movement_params,
            sparsity_level,
            self.modules_not_to_prune,
        )
        new_masks = global_mask_from_threshold(
            movement_params,
            thresh,
            self.modules_not_to_prune,
        )
        return sparsity_level, self.init_params, new_masks

    def snip(self, iter_id, params):
        """Prune weights with smalles grad * weight product/magnitude."""
        # Only implemented for vision datasets with simle cross-entropy loss
        sparsity_level = imp_iter_sparsity(iter_id, self.prune_ratio)
        assert self.dataset_name is not None
        # Get grad weight product on full train set for dataset
        grad_weights = get_snip_grad_weights(
            self.model, self.init_params, self.dataset_name
        )
        thresh = global_magnitude_prune_threshold(
            grad_weights,
            sparsity_level,
            self.modules_not_to_prune,
        )
        new_masks = global_mask_from_threshold(
            grad_weights,
            thresh,
            self.modules_not_to_prune,
        )
        return sparsity_level, self.init_params, new_masks

    def grasp(self, iter_id, params):
        """Prune weights with smallest grad * weight product/magnitude."""
        # Only implemented for vision datasets with simle cross-entropy loss
        sparsity_level = imp_iter_sparsity(iter_id, self.prune_ratio)
        assert self.dataset_name is not None
        # Get grad weight product on full train set for dataset
        grasp_scores = get_grasp_scores(
            self.model, self.init_params, self.dataset_name
        )
        grasp_thresh = global_grasp_prune_threshold(
            grasp_scores, sparsity_level, self.modules_not_to_prune
        )
        new_masks = global_grasp_mask_from_threshold(
            grasp_scores, grasp_thresh, self.modules_not_to_prune
        )
        return sparsity_level, self.init_params, new_masks


def global_magnitude_prune_threshold(
    params, sparsity_level, modules_not_to_prune=None
):
    all_weights = []
    if modules_not_to_prune is None:
        modules_not_to_prune = []

    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(params)).items()
    }

    for m in flat_params:
        # Only consider specific modules and not their biases
        if m not in modules_not_to_prune and not m.endswith("bias"):
            all_weights.append(
                jnp.abs(flat_params[m]).reshape(
                    -1,
                )
            )
    all_magnitudes = jnp.concatenate(all_weights)
    # Assumes sparsity_level in [0, 1]!
    return jnp.percentile(all_magnitudes, sparsity_level * 100)


def global_mask_from_threshold(
    params, threshold, modules_not_to_prune=None, prune_smaller=True
):
    masks = {}
    if modules_not_to_prune is None:
        modules_not_to_prune = []

    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(params)).items()
    }

    for m in flat_params:
        m_key = tuple(m.split("/"))
        if m not in modules_not_to_prune and not m.endswith("bias"):
            if prune_smaller:
                masks[m_key] = (jnp.abs(flat_params[m]) >= threshold).astype(
                    jnp.uint8
                )
            else:
                masks[m_key] = (jnp.abs(flat_params[m]) <= threshold).astype(
                    jnp.uint8
                )
        else:
            masks[m_key] = jnp.ones(flat_params[m].shape).astype(jnp.uint8)
    return FrozenDict(unflatten_dict(masks))


def get_sparsity(params, modules_not_to_prune=None):
    """Calculate the total sparsity and tensor-wise sparsity of params."""
    if modules_not_to_prune is None:
        modules_not_to_prune = []
    total_params = sum(jnp.size(x) for x in jax.tree_leaves(params))
    total_nnz = sum(jnp.sum(x != 0.0) for x in jax.tree_leaves(params))
    leaf_sparsity = jax.tree_map(
        lambda x: jnp.sum(x == 0) / jnp.size(x), params
    )
    prunable_params, pruned_params, prunable_modules = 0, 0, []
    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(params)).items()
    }
    for k in flat_params:
        no_prune = sum([m in k for m in modules_not_to_prune])
        bias = k.endswith("bias")
        if no_prune > 0 or bias:
            pass
        else:
            prunable_params += jnp.size(flat_params[k])
            pruned_params += jnp.sum(flat_params[k] == 0.0)
            prunable_modules.append(k)
    prunable_sparsity = pruned_params / prunable_params
    return (
        total_params,
        total_nnz,
        pruned_params,
        prunable_params,
        prunable_modules,
        leaf_sparsity,
        prunable_sparsity,
    )


def imp_iter_sparsity(iter_id, prune_ratio):
    return 1.0 - (prune_ratio) ** iter_id


def get_final_permuted(rng, masked_init, modules_not_to_prune=None):
    """Loop over layers and create permuted parameter dictionary."""
    if modules_not_to_prune is None:
        modules_not_to_prune = []
    perm_init = {}
    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(masked_init)).items()
    }

    for m in flat_params:
        m_key = tuple(m.split("/"))
        if m not in modules_not_to_prune and not m.endswith("bias"):
            rng, rng_perm = jax.random.split(rng)
            # Get non-zero idx
            nz_idx = flat_params[m] != 0.0
            perm_entries = jax.random.permutation(
                rng_perm, flat_params[m][nz_idx]
            )
            temp = flat_params[m].at[nz_idx].set(perm_entries)
            perm_init[m_key] = temp
        else:
            perm_init[m_key] = flat_params[m]
    return FrozenDict(unflatten_dict(perm_init))


def get_permuted_permuted(rng, masked_init, masks, modules_not_to_prune=None):
    """Loop over layers and create permuted parameter dictionary."""
    if modules_not_to_prune is None:
        modules_not_to_prune = []
    perm_init, perm_masks = {}, {}

    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(masked_init)).items()
    }

    flat_masks = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(masks)).items()
    }

    for m in flat_params:
        m_key = tuple(m.split("/"))
        if m not in modules_not_to_prune and not m.endswith("bias"):
            rng, rng_perm_init, rng_perm_mask = jax.random.split(rng, 3)
            # Get mask & non-zero idx
            mask_idx = flat_masks[m] == 0.0
            perm_mask = jax.random.permutation(rng_perm_mask, mask_idx)
            temp_mask = jnp.ones(flat_params[m].shape).at[perm_mask].set(0)
            perm_masks[m_key] = temp_mask

            nz_idx = flat_params[m] != 0.0
            nzm_idx = temp_mask != 0.0
            perm_entries = jax.random.permutation(
                rng_perm_init, flat_params[m][nz_idx]
            )
            # print(nz_idx[0], perm_idx[0])
            temp_init = flat_params[m].at[nzm_idx].set(perm_entries)
            perm_init[m_key] = temp_init
        else:
            perm_init[m_key] = flat_params[m]
            perm_masks[m_key] = flat_masks[m]

    return FrozenDict(unflatten_dict(perm_init)), FrozenDict(
        unflatten_dict(perm_masks)
    )


def get_random_masks(rng, params, modules_not_to_prune=None):
    """Sample random masks that preserve global sparsity level."""
    if modules_not_to_prune is None:
        modules_not_to_prune = []
    (
        _,
        _,
        pruned_params,
        prunable_params,
        prunable_modules,
        _,
        _,
    ) = get_sparsity(params, modules_not_to_prune)

    flat_params = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(unfreeze(params)).items()
    }

    # Create a placeholder dict for masks of prunable modules
    sub_masks = {}
    for m in flat_params:
        if m in prunable_modules:
            m_key = tuple(m.split("/"))
            sub_masks[m_key] = jnp.ones(flat_params[m].shape).astype(jnp.uint8)

    unflat_masks = unflatten_dict(sub_masks)
    # Init a reshaper that can take flat vector and shape into dict
    reshaper = MaskedParameterReshaper(unflat_masks, verbose=False)
    idx_to_prune = jax.random.choice(
        rng, jnp.arange(reshaper.params_to_opt), (pruned_params,), replace=False
    )
    flat_mask = jnp.ones(reshaper.params_to_opt).at[idx_to_prune].set(0)
    reshaped_flat = reshaper.flat_to_mask(flat_mask)

    reshaped_flat = {
        "/".join(k): v.astype(jnp.float32)
        for k, v in flatten_dict(reshaped_flat).items()
    }

    # Add ones mask for non-prunable modules
    for m in flat_params:
        if m in modules_not_to_prune or m.endswith("bias"):
            reshaped_flat[m] = jnp.ones(flat_params[m].shape)

    re_flat_params = {}
    for m in flat_params:
        m_key = tuple(m.split("/"))
        re_flat_params[m_key] = reshaped_flat[m]

    return FrozenDict(unflatten_dict(re_flat_params))
