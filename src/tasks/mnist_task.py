from typing import Tuple, Optional
import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState
import torch
from torchvision import datasets, transforms


def loss(prediction: chex.Array, target: chex.Array) -> chex.Array:
    target = jax.nn.one_hot(target, 10)
    return -jnp.mean(jnp.sum(prediction * target, axis=1))


def accuracy(prediction: chex.Array, target: chex.Array) -> chex.Array:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)


@dataclass
class MNISTState(TaskState):
    obs: chex.Array
    labels: chex.Array


class MNISTTask(VectorizedTask):
    def __init__(
        self,
        env_name: str = "mnist",
        batch_size: int = 1024,
        test: bool = False,
    ):
        self.max_steps = 1
        self.obs_shape = tuple([28, 28, 1])
        self.act_shape = tuple([10])
        data, labels = get_mnist_data(env_name, test)
        dataloader = BatchLoader(data, labels, batch_size)

        def reset_fn(key: chex.PRNGKey) -> MNISTState:
            if test:
                # Use entire test set for evaluation
                batch_data, batch_labels = data, labels
            else:
                # Sample batch from dataloader
                batch_data, batch_labels = dataloader.sample(key)
            return MNISTState(obs=batch_data, labels=batch_labels)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(
            state: MNISTState, action: chex.Array
        ) -> Tuple[MNISTState, chex.Array, chex.Array]:
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = -loss(action, state.labels)
            return state, reward, jnp.ones(())

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: chex.Array) -> MNISTState:
        return self._reset_fn(key)

    def step(
        self, state: MNISTState, action: chex.Array
    ) -> Tuple[MNISTState, chex.Array, chex.Array]:
        return self._step_fn(state, action)


def get_mnist_data(task_name: str = "mnist", test: bool = False):
    """Get MNIST data via torch and move all data to GPU."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )
    bs = 10000 if test else 60000
    if task_name == "mnist":
        loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", download=True, train=not test, transform=transform
            ),
            batch_size=bs,
            shuffle=False,
        )
    elif task_name == "fmnist":
        loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                "~/data", download=True, train=not test, transform=transform
            ),
            batch_size=bs,
            shuffle=False,
        )
    elif task_name == "kmnist":
        loader = torch.utils.data.DataLoader(
            datasets.KMNIST(
                "~/data", download=True, train=not test, transform=transform
            ),
            batch_size=bs,
            shuffle=False,
        )
    else:
        raise ValueError("Dataset has to mnist, fmnist or kmnist.")
    for _, (data, target) in enumerate(loader):
        break
    return jnp.array(data), jnp.array(target)


class BatchLoader:
    def __init__(
        self,
        X: chex.Array,
        y: Optional[chex.Array],
        batch_size: int,
    ):
        self.X = X
        self.y = y
        self.data_shape = self.X.shape[1:][::-1]
        self.num_train_samples = X.shape[0]
        self.batch_size = batch_size

    def sample(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        """Sample a single batch of X, y data."""
        sample_idx = jax.random.choice(
            key,
            jnp.arange(self.num_train_samples),
            (self.batch_size,),
            replace=False,
        )
        if self.y is None:
            return jnp.take(self.X, sample_idx, axis=0)
        else:
            return (
                jnp.take(self.X, sample_idx, axis=0),
                jnp.take(self.y, sample_idx, axis=0),
            )
