import jax
import jax.numpy as jnp
from src.prune.imp import IMPBaselines
from src.brax_ppo.masked_ppo import masked_ppo_brax
from src.brax_ppo.brax_networks import get_ppo_init_brax
from src.brax_ppo.brax_configs import brax_configs
from src.gym_ppo.masked_ppo import masked_ppo_gymnax
from src.gym_ppo.gym_networks import get_ppo_init_gym


def main(config, log):
    """Run IMP procedure - init & [mask, train, update]x Iterations."""
    # ------------------------- INIT MODEL & TRAINER ------------------------ #
    # Setup task & network apply function & PPO.
    rng = jax.random.PRNGKey(config.seed_id)
    if config.env_name != "Pendulum-v1":
        policy, params_init, input_dim = get_ppo_init_brax(
            rng, config.env_name, **config.model_config
        )
        ppo_config = brax_configs[config.env_name]
    else:
        policy, params_init, input_dim = get_ppo_init_gym(
            rng, config.env_name, **config.model_config
        )
        ppo_config = config.ppo_config
    # -------------------------- GET FIRST IMP MASK ------------------------- #
    imp_baselines = IMPBaselines(
        **config.imp_config,
        init_params=params_init,
        model=policy,
        fake_args=jnp.zeros(input_dim),
    )
    (
        sparsity_level,
        params_to_train,
        masks,
        summary,
        thresh,
    ) = imp_baselines.apply(rng, 0, params_init)

    for iter_id in range(1, config.num_imp_iters + 1):
        # ---------------------- FIX RANDOMNESS FOR ITER -------------------- #
        rng, rng_mask = jax.random.split(rng)

        # ---------------------------- RUN PPO TRAINING ---------------------- #
        if config.env_name != "Pendulum-v1":
            final_ckpt, final_perf, best_ckpt, best_perf = masked_ppo_brax(
                iter_id,
                masks,
                params_to_train,
                ppo_config,
                config,
                log,
            )
        else:
            final_ckpt, final_perf, best_ckpt, best_perf = masked_ppo_gymnax(
                iter_id,
                masks,
                params_to_train,
                policy,
                ppo_config,
                config,
                config.seed_id,
                log,
            )

        # ------------------------- UPDATE MLE LOGGER ----------------------- #
        # Save meta log/networks from IMP iteration
        log.update(
            {"imp_iter": iter_id},
            {
                "final_perf": final_perf,
                "best_perf": best_perf,
                "sparsity": sparsity_level,
                "trainable_params": summary[1],
                "pruned_params": summary[2],
                "prunable_params": summary[3],
                "prune_threshold": float(thresh),
            },
            save=True,
        )
        networks_to_store = {
            "init": params_init,
            "mask": masks,
            "final": final_ckpt,
            "best": best_ckpt,
        }
        log.save_extra(
            networks_to_store,
            f"seed_{config.seed_id}/imp_nets_{iter_id}.pkl",
        )

        # ---------------------- UPDATE LTH MASK/WEIGHTS -------------------- #
        # Update the mask based on pruning ratio and magnitudes
        if config.network_to_prune == "final":
            pruning_ckpt = final_ckpt
        elif config.network_to_prune == "best":
            pruning_ckpt = best_ckpt
        else:
            raise ValueError("Pune based on best of final checkpoint.")

        (
            sparsity_level,
            params_to_train,
            masks,
            summary,
            thresh,
        ) = imp_baselines.apply(rng_mask, iter_id, pruning_ckpt)

        # Check sparsity level correctness
        assert jnp.isclose(sparsity_level, summary[-1], atol=1e-02)


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/ppo.yaml")
    main(mle.train_config, mle.log)
