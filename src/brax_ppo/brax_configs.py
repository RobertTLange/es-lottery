# Taken from brax demo notebook for PPO.
# Only provide SAC for walker2d and hopper.
# NOTE(RobertTLange): Doubled training timesteps

ant = {
    "num_timesteps": 50_000_000,
    "num_evals": 20,
    "reward_scaling": 10,
    "episode_length": 500,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 5,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-2,
    "num_envs": 2048,
    "batch_size": 1024,
}

grasp = {
    "num_timesteps": 1_200_000_000,
    "num_evals": 10,
    "reward_scaling": 10,
    "episode_length": 500,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 20,
    "num_minibatches": 32,
    "num_updates_per_batch": 2,
    "discounting": 0.99,
    "learning_rate": 3e-4,
    "entropy_cost": 0.001,
    "num_envs": 2048,
    "batch_size": 256,
}

fetch = {
    "num_timesteps": 200_000_000,
    "num_evals": 20,
    "reward_scaling": 5,
    "episode_length": 500,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 20,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.997,
    "learning_rate": 3e-4,
    "entropy_cost": 0.001,
    "num_envs": 2048,
    "batch_size": 256,
}

halfcheetah = {
    "num_timesteps": 200_000_000,
    "num_evals": 20,
    "reward_scaling": 1,
    "episode_length": 500,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 20,
    "num_minibatches": 32,
    "num_updates_per_batch": 8,
    "discounting": 0.95,
    "learning_rate": 3e-4,
    "entropy_cost": 0.001,
    "num_envs": 2048,
    "batch_size": 512,
}

humanoid = {
    "num_timesteps": 100_000_000,
    "num_evals": 20,
    "reward_scaling": 0.1,
    "episode_length": 500,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 10,
    "num_minibatches": 32,
    "num_updates_per_batch": 8,
    "discounting": 0.97,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-3,
    "num_envs": 2048,
    "batch_size": 1024,
}

reacher = {
    "num_timesteps": 200_000_000,
    "num_evals": 20,
    "reward_scaling": 5,
    "episode_length": 500,
    "normalize_observations": True,
    "action_repeat": 4,
    "unroll_length": 50,
    "num_minibatches": 32,
    "num_updates_per_batch": 8,
    "discounting": 0.95,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-3,
    "num_envs": 2048,
    "batch_size": 256,
}

ur5e = {
    "num_timesteps": 40_000_000,
    "num_evals": 20,
    "reward_scaling": 10,
    "episode_length": 500,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 5,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.95,
    "learning_rate": 2e-4,
    "entropy_cost": 1e-2,
    "num_envs": 2048,
    "batch_size": 1024,
}

hopper = {
    "num_timesteps": 200_000_000,
    "num_evals": 20,
    "reward_scaling": 5,
    "episode_length": 500,
    "normalize_observations": True,
    "action_repeat": 2,
    "unroll_length": 20,
    "num_minibatches": 32,
    "num_updates_per_batch": 8,
    "discounting": 0.999,
    "learning_rate": 6e-4,
    "entropy_cost": 1e-2,
    "num_envs": 2048,
    "batch_size": 256,
}

walker2d = {
    "num_timesteps": 200_000_000,
    "num_evals": 20,
    "reward_scaling": 10,
    "episode_length": 500,
    "normalize_observations": True,
    "action_repeat": 3,
    "unroll_length": 50,
    "num_minibatches": 32,
    "num_updates_per_batch": 8,
    "discounting": 0.98,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-2,
    "num_envs": 2048,
    "batch_size": 256,
}


brax_configs = {
    "ant": ant,
    "fetch": fetch,
    "grasp": grasp,
    "halfcheetah": halfcheetah,
    "hopper": hopper,
    "humanoid": humanoid,
    "reacher": reacher,
    "ur5e": ur5e,
    "walker2d": walker2d,
}
