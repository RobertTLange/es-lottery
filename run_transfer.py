from mle_logging.utils import load_pkl_object
import os
import jax
import jax.numpy as jnp
from src.es.masked_es import masked_es
from src.mnist_gd.train_mnist import masked_sgd_mnist
from src.brax_ppo.masked_ppo import masked_ppo_brax
from src.brax_ppo.brax_networks import get_ppo_init_brax
from src.brax_ppo.brax_configs import brax_configs
from src.bench_tasks import get_evojax_task


def main(config, log):
    """Run transfer procedure - init & [mask, train, update]x Iterations."""
    # ------------------------- INIT MODEL & TRAINER ------------------------ #
    # Setup task & network apply function & ES.
    train_task, test_task, policy = get_evojax_task(
        config.env_name, **config.task_config, **config.model_config
    )

    for iter_id in range(1, config.num_imp_iters + 1):
        # -------------------------- GET FIRST IMP MASK --------------------- #
        # Get all ticket paths - and load first one
        ticket_path = os.path.join(
            config.ticket_dir,
            f"seed_{str(config.seed_id)}",
            f"imp_nets_{iter_id}.pkl",
        )
        lth_ckpt = load_pkl_object(ticket_path)
        masks = lth_ckpt["mask"]
        params_to_train = lth_ckpt["init"]
        print(f"Loaded: {ticket_path}")
        # ------------------------ RUN ES/PPO TRAINING ---------------------- #
        if config.train_type == "ES":
            (
                final_ckpt,
                final_perf,
                final_sigma,
                best_ckpt,
                best_perf,
                best_sigma,
            ) = masked_es(
                iter_id,
                masks,
                params_to_train,
                config,
                train_task,
                test_task,
                policy,
                log,
                config.fixed_seed,
            )
        elif config.train_type == "SGD-mnist":
            final_ckpt, final_perf, best_ckpt, best_perf = masked_sgd_mnist(
                iter_id,
                masks,
                params_to_train,
                config,
                policy,
                log,
            )
        elif config.train_type == "SGD-ppo":
            ppo_config = brax_configs[config.env_name]
            _, params_init, _ = get_ppo_init_brax(
                jax.random.PRNGKey(0), config.env_name, **config.model_config
            )

            def set_val(x, y):
                if len(y.shape) == 1:
                    z = x.at[: y.shape[0]].set(y[:])
                else:
                    z = x.at[:, : y.shape[1]].set(y[:, : y.shape[1]])
                return z

            params_to_train = jax.tree_map(
                lambda x, y: set_val(x, y),
                params_init,
                params_to_train,
            )
            masks_raw = jax.tree_map(
                lambda x: jnp.ones(x.shape),
                params_init,
            )

            masks = jax.tree_map(
                lambda x, y: set_val(x, y),
                masks_raw,
                masks,
            )

            final_ckpt, final_perf, best_ckpt, best_perf = masked_ppo_brax(
                iter_id,
                masks,
                params_to_train,
                ppo_config,
                config,
                log,
            )

        # ------------------------- UPDATE MLE LOGGER ----------------------- #
        # Save meta log/networks from IMP iteration
        log.update(
            {"imp_iter": iter_id},
            {
                "final_perf": float(final_perf),
                "best_perf": float(best_perf),
            },
            save=True,
        )
        networks_to_store = {
            "final_mean": final_ckpt,
            "best_mean": best_ckpt,
            # "final_sigma": final_sigma,
            # "best_sigma": best_sigma,
        }
        log.save_extra(
            networks_to_store,
            f"seed_{config.seed_id}/imp_nets_{iter_id}.pkl",
        )


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/transfer/mnist_es2es.yaml")
    main(mle.train_config, mle.log)
