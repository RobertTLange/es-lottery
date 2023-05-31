"""Run masked EvoJAX evolution loop."""
import numpy as np
import jax.numpy as jnp
from evosax import Strategies
from evojax.obs_norm import ObsNormalizer
from evojax.sim_mgr import SimManager
from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper
from src.es.tuned_hparams import get_tuned_hparams
from src.prune.masked_reshaper import MaskedParameterReshaper, apply_mask


def masked_es(
    imp_iter,
    masks,
    params_to_train,
    config,
    train_task,
    test_task,
    policy,
    log,
    fixed_seed=False,
):
    """Running an ES loop on Evojax task with evosax strategy."""
    # Setup flat masked parameter reshaper
    masked_reshaper = MaskedParameterReshaper(masks)
    masked_params = apply_mask(params_to_train, masks)
    flat_params = masked_reshaper.masked_to_flat_sparse(masked_params)

    # Get tuned es_config from benchmark data
    tuned_config = get_tuned_hparams(config.strategy_name, config.env_name)
    es_config = {**tuned_config, **config.es_config}
    solver = Evosax2JAX_Wrapper(
        Strategies[config.strategy_name],
        param_size=flat_params.shape[0],
        pop_size=config.popsize,
        es_config=es_config,
        seed=config.seed_id + imp_iter * (1 - fixed_seed),
    )
    obs_normalizer = ObsNormalizer(
        obs_shape=train_task.obs_shape, dummy=not config.normalize_obs
    )
    sim_mgr = SimManager(
        policy_net=policy,
        train_vec_task=train_task,
        valid_vec_task=test_task,
        seed=config.seed_id + imp_iter * (1 - fixed_seed),
        obs_normalizer=obs_normalizer,
        pop_size=config.popsize,
        use_for_loop=False,
        **config.eval_config,
    )

    # Set the mean to the non-sparse initialization
    solver.es_state = solver.es_state.replace(mean=flat_params)

    # Track best test performance and corresponding ckpt
    best_perf = -jnp.finfo(jnp.float32).max
    params_best = masked_reshaper.flat_to_mask(solver.es_state.mean)
    if config.strategy_name == "Sep_CMA_ES":
        sigma_best = masked_reshaper.flat_to_sigma(
            solver.es_state.sigma * solver.es_state.C
        )
    elif config.strategy_name in ["SNES", "PGPE", "DES"]:
        sigma_best = masked_reshaper.flat_to_sigma(solver.es_state.sigma)
    else:
        raise ValueError("Unknown strategy")
    print(f"START EVOLVING {flat_params.shape[0]} PARAMS.")
    # Run ES Loop.
    for gen_counter in range(config.num_generations):
        params = solver.ask()
        params_mask = masked_reshaper.to_mask(params)
        params_flat = masked_reshaper.to_flat(params_mask)
        scores, _ = sim_mgr.eval_params(params=params_flat, test=False)
        solver.tell(fitness=scores)
        if gen_counter == 0 or (gen_counter + 1) % config.eval_every_gen == 0:
            params_best = masked_reshaper.flat_to_mask(solver.best_params)
            params_best_flat = masked_reshaper.mask_to_flat(params_best)
            test_scores, _ = sim_mgr.eval_params(
                params=params_best_flat, test=True
            )

            if config.strategy_name == "Sep_CMA_ES":
                sigma_log = masked_reshaper.flat_to_sigma(solver.es_state.C)
            elif config.strategy_name in ["SNES", "PGPE", "DES"]:
                sigma_log = masked_reshaper.flat_to_sigma(solver.es_state.sigma)
            else:
                raise ValueError("Unknown strategy")
            model_to_log = {
                "mean": masked_reshaper.flat_to_mask(solver.es_state.mean),
                "sigma": sigma_log,
            }
            log.update(
                {"imp_iter": imp_iter, "num_gens": gen_counter + 1},
                {
                    "train_perf": float(np.nanmean(scores)),
                    "test_perf": float(np.nanmean(test_scores)),
                },
                model=model_to_log,
                save=True,
            )

            # Update best performance tracker
            if float(np.nanmean(test_scores)) > best_perf:
                best_perf = float(np.nanmean(test_scores))
                params_best = masked_reshaper.flat_to_mask(solver.es_state.mean)
                if config.strategy_name == "Sep_CMA_ES":
                    sigma_best = masked_reshaper.flat_to_sigma(
                        solver.es_state.sigma * solver.es_state.C
                    )
                elif config.strategy_name in ["SNES", "PGPE", "DES"]:
                    sigma_best = masked_reshaper.flat_to_sigma(
                        solver.es_state.sigma
                    )
                else:
                    raise ValueError("Unknown strategy")
    params_final = masked_reshaper.flat_to_mask(solver.es_state.mean)

    if config.strategy_name == "Sep_CMA_ES":
        sigma_final = masked_reshaper.flat_to_sigma(solver.es_state.C)
    elif config.strategy_name in ["SNES", "PGPE", "DES"]:
        sigma_final = masked_reshaper.flat_to_sigma(solver.es_state.sigma)
    else:
        raise ValueError("Unknown strategy")
    perf_final = float(np.nanmean(test_scores))
    return (
        params_final,
        perf_final,
        sigma_final,
        params_best,
        best_perf,
        sigma_best,
    )
