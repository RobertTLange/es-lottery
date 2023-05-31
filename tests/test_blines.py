import jax
import jax.numpy as jnp
from src.imp import IMPBaselines, apply_mask
from evosax_benchmark.evojax_tasks import get_evojax_task


def test_final_ticket():
    rng = jax.random.PRNGKey(0)
    _, _, policy = get_evojax_task("ant")
    params_model = policy.model.init(rng, jnp.zeros(policy.input_dim))
    imp_baselines = IMPBaselines(
        "final-ticket",
        params_model,
        0.8,
        [],
        policy,
        jnp.zeros(policy.input_dim),
        verbose=False,
    )

    for i in range(20):
        sparsity_level, params_to_train, masks, summary = imp_baselines.apply(
            rng, i, params_model
        )
        assert jnp.isclose(sparsity_level, summary[-1], atol=1e-03)
        params_model = apply_mask(params_to_train, masks)
        print(i, sparsity_level, summary[-1])


def test_final_permute():
    rng = jax.random.PRNGKey(0)
    _, _, policy = get_evojax_task("ant")
    params_model = policy.model.init(rng, jnp.zeros(policy.input_dim))

    imp_baselines = IMPBaselines(
        "final-ticket_permute",
        params_model,
        0.8,
        [],
        policy,
        jnp.zeros(policy.input_dim),
        verbose=False,
    )
    for i in range(20):
        sparsity_level, params_to_train, masks, summary = imp_baselines.apply(
            rng, i, params_model
        )
        assert jnp.isclose(sparsity_level, summary[-1], atol=1e-03)
        params_model = apply_mask(params_to_train, masks)
        print(i, sparsity_level, summary[-1])


def test_final_permute_ticket_permute():
    rng = jax.random.PRNGKey(0)
    _, _, policy = get_evojax_task("ant")
    params_model = policy.model.init(rng, jnp.zeros(policy.input_dim))

    imp_baselines = IMPBaselines(
        "final_permute-ticket_permute",
        params_model,
        0.8,
        [],
        policy,
        jnp.zeros(policy.input_dim),
        verbose=False,
    )

    for i in range(20):
        sparsity_level, params_to_train, masks, summary = imp_baselines.apply(
            rng, i, params_model
        )
        assert jnp.isclose(sparsity_level, summary[-1], atol=1e-03)
        params_model = apply_mask(params_to_train, masks)
        print(i, sparsity_level, summary[-1])


def test_random_reinit():
    rng = jax.random.PRNGKey(0)
    _, _, policy = get_evojax_task("ant")
    params_model = policy.model.init(rng, jnp.zeros(policy.input_dim))

    imp_baselines = IMPBaselines(
        "random-reinit",
        params_model,
        0.8,
        [],
        policy.model,
        jnp.zeros(policy.input_dim),
        verbose=False,
    )

    for i in range(20):
        sparsity_level, params_to_train, masks, summary = imp_baselines.apply(
            rng, i, params_model
        )
        assert jnp.isclose(sparsity_level, summary[-1], atol=1e-03)
        params_model = apply_mask(params_to_train, masks)
        print(i, sparsity_level, summary[-1])


if __name__ == "__main__":
    # test_final_ticket()
    # test_final_permute()
    # test_final_permute_ticket_permute()
    test_random_reinit()
