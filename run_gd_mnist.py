import jax
import jax.numpy as jnp
from src.prune.imp import IMPBaselines
from src.mnist_gd.train_mnist import masked_sgd_mnist
from src.bench_tasks import get_evojax_task


def main(config, log):
    """Run IMP procedure - init & [mask, train, update]x Iterations."""
    # ------------------------- INIT MODEL & TRAINER ------------------------ #
    # Setup task & network apply function & ES.
    rng = jax.random.PRNGKey(config.seed_id)
    train_task, test_task, policy = get_evojax_task(
        config.env_name, **config.task_config, **config.model_config
    )
    params_init = policy.model.init(rng, jnp.zeros(policy.input_dim))
    # -------------------------- GET FIRST IMP MASK ------------------------- #
    imp_baselines = IMPBaselines(
        **config.imp_config,
        init_params=params_init,
        model=policy.model,
        fake_args=jnp.zeros(policy.input_dim),
        dataset_name=config.env_name,
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

        # ---------------------------- RUN ES TRAINING ---------------------- #
        (final_ckpt, final_perf, best_ckpt, best_perf,) = masked_sgd_mnist(
            iter_id,
            masks,
            params_to_train,
            config,
            policy,
            log,
        )

        # ------------------------- UPDATE MLE LOGGER ----------------------- #
        # Save meta log/networks from IMP iteration
        log.update(
            {"imp_iter": iter_id},
            {
                "final_perf": float(final_perf),
                "best_perf": float(best_perf),
                "sparsity": float(sparsity_level),
                "trainable_params": float(summary[1]),
                "pruned_params": float(summary[2]),
                "prunable_params": float(summary[3]),
                "prune_threshold": float(thresh),
            },
            save=True,
        )
        networks_to_store = {
            "init": params_init,
            "mask": masks,
            "final_mean": final_ckpt,
            "best_mean": best_ckpt,
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
    mle = MLExperiment(config_fname="configs/sgd.yaml")
    main(mle.train_config, mle.log)
