from src.gym_ppo.gym_ppo import train


def masked_ppo_gymnax(
    iter_id, masks, params_to_train, policy, ppo_config, config, seed_id, log
):
    """Run masked PPO training."""
    final_ckpt, final_perf, best_ckpt, best_perf = train(
        iter_id, masks, params_to_train, policy, ppo_config, seed_id, log
    )
    return final_ckpt, final_perf, best_ckpt, best_perf
