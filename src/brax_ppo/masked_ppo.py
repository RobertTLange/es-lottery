from .brax_ppo import train
from brax.v1 import envs


def masked_ppo_brax(iter_id, masks, params_to_train, ppo_config, config, log):
    """Run masked PPO training."""

    def progress(num_steps, imp_iter, metrics, model):
        log.update(
            {"imp_iter": imp_iter, "num_steps": num_steps},
            {
                "test_perf": float(metrics["eval/episode_reward"]),
            },
            model=model,
            save=True,
        )

    env = envs.get_environment(env_name=config.env_name, legacy_spring=True)

    phidden_sizes = (
        config.model_config.hidden_dims,
    ) * config.model_config.hidden_layers
    final_ckpt, final_perf, best_ckpt, best_perf = train(
        policy_params_to_train=params_to_train,
        policy_masks=masks,
        environment=env,
        progress_fn=progress,
        seed=iter_id,
        phidden_sizes=phidden_sizes,
        **ppo_config,
    )
    return final_ckpt, final_perf, best_ckpt, best_perf
