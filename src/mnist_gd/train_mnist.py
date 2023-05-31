import torch
from torchvision import datasets, transforms
import numpy as np
import optax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


def apply_mask(params, masks):
    return jax.tree_util.tree_map(lambda x, y: x * y, params, masks)


def masked_sgd_mnist(iter_id, masks, params, train_config, model, log):
    rng = jax.random.PRNGKey(train_config.seed_id + iter_id)
    tx = optax.adam(train_config.lrate)
    masked_params = apply_mask(params, masks)
    train_state = TrainState.create(
        apply_fn=model.model.apply,
        params=masked_params,
        tx=tx,
    )

    @jax.jit
    def apply_model(rng, state, images, labels):
        """Compute grads, loss and accuracy (single batch)."""

        def loss_fn(params):
            logits = state.apply_fn(params, images)
            one_hot = jax.nn.one_hot(labels, 10)
            # l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(state.params))
            cent_loss = optax.softmax_cross_entropy(
                logits=logits, labels=one_hot
            )
            loss = jnp.mean(cent_loss)  # + train_config["w_decay"] * l2_loss)
            return loss, (logits, state)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        logits, _ = aux[1]
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return grads, aux, accuracy

    @jax.jit
    def update_model(state, grads, masks):
        masked_grads = apply_mask(grads, masks)
        return state.apply_gradients(grads=masked_grads)

    def eval_step(rng, state, image, labels):
        logits = state.apply_fn(state.params, image)
        one_hot = jax.nn.one_hot(labels, 10)
        cent_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        loss = jnp.mean(cent_loss)
        acc = jnp.mean(jnp.argmax(logits, -1) == labels)
        return loss, acc

    best_perf_yet, best_model_ckpt = -jnp.inf, None
    if train_config.env_name == "mnist":
        train_loader, test_loader = get_mnist_loaders(train_config.batch_size)
    elif train_config.env_name == "fmnist":
        train_loader, test_loader = get_fmnist_loaders(train_config.batch_size)
    elif train_config.env_name == "kmnist":
        train_loader, test_loader = get_kmnist_loaders(train_config.batch_size)

    for step in range(train_config.num_epochs):
        # Loop over training epochs and batches
        epoch_loss, epoch_acc = [], []
        for batch_idx, (data, target) in enumerate(train_loader):
            rng, rng_step, rng_eval = jax.random.split(rng, 3)
            batch_images = jnp.array(data)
            batch_labels = jnp.array(target)
            grads, aux, acc = apply_model(
                rng_step, train_state, batch_images, batch_labels
            )
            train_state = update_model(train_state, grads, masks)
            epoch_loss.append(aux[0])
            epoch_acc.append(acc)
            step += 1

        # Mean training loss/acc and compute test performance
        train_loss = np.mean(epoch_loss)
        train_acc = np.mean(epoch_acc)

        test_loss, test_acc = 0, 0
        for batch_idx, (data, target) in enumerate(test_loader):
            batch_images = jnp.array(data)
            batch_labels = jnp.array(target)
            test_perf = eval_step(
                rng_eval, train_state, batch_images, batch_labels
            )
            test_loss = test_perf[0]
            test_acc = test_perf[1]

        if test_acc > best_perf_yet:
            best_perf_yet = test_acc
            best_model_ckpt = train_state.params

        log.update(
            {"imp_iter": iter_id, "num_epochs": step + 1},
            {"test_perf": test_acc},
            model=train_state.params,
            save=True,
        )
    return (
        train_state.params,
        test_acc,
        best_model_ckpt,
        best_perf_yet,
    )


def get_mnist_loaders(batch_size: int):
    """Get PyTorch Data Loaders for MNIST tasks."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "~/data", download=True, train=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    testloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "~/data", download=True, train=False, transform=transform
        ),
        batch_size=10000,
        shuffle=False,
    )

    return trainloader, testloader


def get_fmnist_loaders(batch_size: int):
    """Get PyTorch Data Loaders for F-MNIST tasks."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    trainloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "~/data", download=True, train=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    testloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "~/data", download=True, train=False, transform=transform
        ),
        batch_size=10000,
        shuffle=False,
    )

    return trainloader, testloader


def get_kmnist_loaders(batch_size: int):
    """Get PyTorch Data Loaders for K-MNIST tasks."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    trainloader = torch.utils.data.DataLoader(
        datasets.KMNIST(
            "~/data", download=True, train=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    testloader = torch.utils.data.DataLoader(
        datasets.KMNIST(
            "~/data", download=True, train=False, transform=transform
        ),
        batch_size=10000,
        shuffle=False,
    )

    return trainloader, testloader
