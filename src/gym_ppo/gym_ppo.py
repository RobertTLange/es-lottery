from functools import partial
import optax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from src.gym_ppo.gym_data import RolloutManager, BatchManager, update


def apply_mask(params, masks):
    return jax.tree_util.tree_map(lambda x, y: x * y, params, masks)


def train(iter_id, masks, params, model, config, seed_id, mle_log):
    """Training loop for PPO based on https://github.com/bmazoure/ppo_jax."""
    best_perf = -1e10
    rng = jax.random.PRNGKey(iter_id + seed_id)
    num_total_epochs = int(config.num_train_steps // config.num_train_envs + 1)
    num_steps_warm_up = int(config.num_train_steps * config.lr_warmup)
    schedule_fn = optax.linear_schedule(
        init_value=-float(config.lr_begin),
        end_value=-float(config.lr_end),
        transition_steps=num_steps_warm_up,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(schedule_fn),
    )

    masked_params = apply_mask(params, masks)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=masked_params,
        tx=tx,
    )
    # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager = RolloutManager(model, config.env_name, {}, {})

    batch_manager = BatchManager(
        discount=config.gamma,
        gae_lambda=config.gae_lambda,
        n_steps=config.n_steps + 1,
        num_envs=config.num_train_envs,
        action_size=rollout_manager.action_size,
        state_space=rollout_manager.observation_space,
    )

    @partial(jax.jit, static_argnums=5)
    def get_transition(
        train_state: TrainState,
        obs: jnp.ndarray,
        state: dict,
        batch,
        rng: jax.random.PRNGKey,
        num_train_envs: int,
    ):
        action, log_pi, value, new_key = rollout_manager.select_action(
            train_state, obs, rng
        )
        # print(action.shape)
        new_key, key_step = jax.random.split(new_key)
        b_rng = jax.random.split(key_step, num_train_envs)
        # Automatic env resetting in gymnax step!
        next_obs, next_state, reward, done, _ = rollout_manager.batch_step(
            b_rng, state, action
        )
        batch = batch_manager.append(
            batch, obs, action, reward, done, log_pi, value
        )
        return train_state, next_obs, next_state, batch, new_key

    batch = batch_manager.reset()

    rng, rng_step, rng_reset, rng_eval, rng_update = jax.random.split(rng, 5)
    obs, state = rollout_manager.batch_reset(
        jax.random.split(rng_reset, config.num_train_envs)
    )

    total_steps = 0
    log_steps, log_return = [], []
    for step in range(1, num_total_epochs + 1):
        train_state, obs, state, batch, rng_step = get_transition(
            train_state,
            obs,
            state,
            batch,
            rng_step,
            config.num_train_envs,
        )
        total_steps += config.num_train_envs
        if step % (config.n_steps + 1) == 0:
            metric_dict, train_state, rng_update = update(
                train_state,
                batch_manager.get(batch),
                config.num_train_envs,
                config.n_steps,
                config.n_minibatch,
                config.epoch_ppo,
                config.clip_eps,
                config.entropy_coeff,
                config.critic_coeff,
                rng_update,
                masks,
            )
            batch = batch_manager.reset()

        if (step + 1) % config.evaluate_every_epochs == 0:
            rng, rng_eval = jax.random.split(rng)
            rewards = rollout_manager.batch_evaluate(
                rng_eval,
                train_state,
                config.num_test_rollouts,
            )
            log_steps.append(total_steps)
            log_return.append(rewards)

            if rewards > best_perf:
                best_perf = rewards
                best_ckpt = train_state.params

            mle_log.update(
                {"num_steps": total_steps},
                {"test_perf": rewards},
                # model=train_state.params,
                save=True,
            )

    final_ckpt = train_state.params
    final_perf = rewards
    return (final_ckpt, final_perf, best_ckpt, best_perf)
