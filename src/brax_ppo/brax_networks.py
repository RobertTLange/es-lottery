# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO networks."""

from typing import Sequence, Tuple

from brax.v1 import envs
from brax.training import distribution

# from brax.training import networks
import src.brax_ppo.brax_nets as networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen


@flax.struct.dataclass
class PPONetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPONetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = (
            ppo_networks.parametric_action_distribution
        )

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = policy_network.apply(*params, observations)
            if deterministic:
                return (
                    ppo_networks.parametric_action_distribution.mode(logits),
                    {},
                )
            raw_actions = (
                parametric_action_distribution.sample_no_postprocessing(
                    logits, key_sample
                )
            )
            log_prob = parametric_action_distribution.log_prob(
                logits, raw_actions
            )
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return policy

    return make_policy


def make_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,)
    * 2,  # NOTE(RobertTLange): Changed from 4 to 2 layers form comparison
    value_hidden_layer_sizes: Sequence[int] = (256,)
    * 5,  # NOTE(RobertTLange): Don't prune critic for comparison
    activation: networks.ActivationFn = linen.swish,
    # NOTE(RobertTLange): Change from swish to tanh for comparison?
) -> PPONetworks:
    """Make PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def get_ppo_init_brax(
    rng,
    env_name: str,
    hidden_layers: int,
    hidden_dims: int,
    activation_fn: str = "swish",
):
    """Initialize a dense policy network for PPO training."""
    env = envs.get_environment(env_name=env_name, legacy_spring=True)
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=env.action_size
    )
    if activation_fn == "swish":
        activation = linen.swish
    elif activation_fn == "tanh":
        activation = linen.tanh

    policy_hidden_layer_sizes = (hidden_dims,) * hidden_layers
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        env.observation_size,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
    )
    params_init = policy_network.init(rng)
    return policy_network, params_init, (1, env.observation_size)
