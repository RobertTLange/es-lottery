import jax
import jax.numpy as jnp
import optax
import torch
from torchvision import datasets, transforms


def get_snip_grad_weights(
    model, params, dataset_name: str, batch_size: int = 60000
):
    """Calculate gradient-weight product for each parameter and prune."""
    X, y = get_data_loaders(dataset_name, batch_size=batch_size)

    @jax.jit
    def get_grads(params, images, labels):
        """Compute grads, loss and accuracy (single batch)."""

        def loss_fn(params):
            logits = model.apply(params, images)
            one_hot = jax.nn.one_hot(labels, 10)
            cent_loss = optax.softmax_cross_entropy(
                logits=logits, labels=one_hot
            )
            loss = jnp.mean(cent_loss)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)
        return grads

    grads = get_grads(params, X, y)
    grad_weights = jax.tree_util.tree_map(lambda x, y: x * y, params, grads)
    return grad_weights


def get_data_loaders(dataset_name: str, batch_size: int = 60000):
    """Get PyTorch Data Loaders for MNIST tasks."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    if dataset_name == "mnist":
        trainloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", download=True, train=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    elif dataset_name == "fmnist":
        trainloader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                "~/data", download=True, train=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    elif dataset_name == "kmnist":
        trainloader = torch.utils.data.DataLoader(
            datasets.KMNIST(
                "~/data", download=True, train=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    for batch_idx, (data, target) in enumerate(trainloader):
        X, y = data, target
        break
    return jnp.array(X), jnp.array(y)
