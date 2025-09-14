from typing import Iterator, Tuple, Union
import argparse

import mlx.core as mx
from mlx.nn.losses import cross_entropy
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from src.transformer import DecoderTransformer
from src.utils import load_config_basic


class StoryDataset:
    def __init__(
        self, data: str, batch_size: int = 32, seq_len: int = 128, shuffle: bool = True
    ):
        chars = sorted(list(set(data)))
        # Here using a [BOS] token doesn't make sense because we are training from the corpus by grabbing parts of a very big text (so majority of times not beggining of sentences)

        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size} tokens and the vocab size is {vocab_size}")

        self.stoi = {char: num for num, char in enumerate(chars)}
        self.itos = {num: char for num, char in enumerate(chars)}

        self.vocab_size = vocab_size
        self.data = data
        self.seq_len = seq_len

        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(
            0, len(self.data) - self.seq_len
        )  # last char has no target

    def __reset__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)
        self._current = 0

    def __iter__(self) -> Iterator[Tuple[mx.array, mx.array]]:
        """Return the iterator object (self)."""
        self.__reset__()
        return self

    def __next__(self) -> Tuple[mx.array, mx.array]:
        """Return the next mini-batch."""
        if self._current + self.batch_size > len(self._indices):
            raise StopIteration

        # Slice indices for the current batch
        idxs = self._indices[self._current : self._current + self.batch_size]

        batch_x, batch_y = [], []
        for start in idxs:
            chunk = self.data[start : start + seq_len + 1]
            dix = [self.stoi[s] for s in chunk]

            batch_x.append(dix[:-1])
            batch_y.append(dix[1:])

        self._current += self.batch_size
        return mx.array(batch_x), mx.array(batch_y)

    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array]:
        # grab a chunk of (seq_len + 1) characters from the data
        chunk = self.data[idx : idx + self.seq_len + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as mx.arrays
        x = mx.array(dix[:-1])
        y = mx.array(dix[1:])
        return x, y

    def __len__(self) -> int:
        return int(np.ceil((len(self.data) - self.seq_len) / self.batch_size))


def train(
    model: DecoderTransformer,
    train_dataset: StoryDataset,
    lr: float,
    epochs: int,
    batch_size: int,
    seq_len: int,
    use_aux_loss: bool = False,
    expert_level_balance: float = 0.01,  # Weight for load balancing loss
):
    def loss_fn(model, x, y, use_aux_loss: bool = False):
        if use_aux_loss:
            out, load_balance_loss = model(x, return_aux_loss=use_aux_loss)
            cross_entropy_loss = nn.losses.cross_entropy(out, y, reduction="mean")
            total_loss = cross_entropy_loss + expert_level_balance * load_balance_loss
            return total_loss, (cross_entropy_loss, load_balance_loss, out)
        else:
            out = model(x)
            cross_entropy_loss = nn.losses.cross_entropy(out, y, reduction="mean")
            return cross_entropy_loss, (cross_entropy_loss, mx.array(0.0), out)

    # Get a function which gives the loss and gradient of the
    # loss with respect to the model's trainable parameters
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    optimizer = optim.Adam(learning_rate=lr)

    losses, accuracies = [], []

    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracies = 0.0

        with tqdm(
            train_dataset,
            total=len(train_dataset),
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
        ) as pbar:
            for step, (x_batch, y_batch) in enumerate(pbar):
                x_batch = mx.expand_dims(x_batch, -1)

                # Don't know how to get the model outputs and also the loss
                (loss, (main_loss, aux_loss, out)), grads = loss_and_grad_fn(model, x_batch, y_batch, use_aux_loss)

                optimizer.update(model, grads)

                # Force a graph evaluation
                mx.eval(model.parameters(), optimizer.state)

                accuracy = mx.softmax(out).argmax(axis=-1) == y_batch  # (batch_size, seq_len)

                total_loss += loss
                total_accuracies += accuracy.sum()

                pbar.set_postfix(
                    loss=f"{loss}",
                    acc=f"{accuracy.mean()}",
                )

        # Average over the entire dataset
        losses.append(total_loss / len(train_dataset))
        accuracies.append(
            total_accuracies / (len(train_dataset) * batch_size * seq_len)
        )

        print(f"Loss: {losses[epoch]}, Accuracy: {accuracies[epoch]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train / run a tiny Transformer written in MLX"
    )

    # misc
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved .safetensors file or to save tensor file",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to .yaml config")
    parser.add_argument("--train", action="store_true", help="If set will train")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = load_config_basic(args.config)

    lr = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    seq_len = config["training"]["sequence_length"]
    expert_level_balance = config["training"].get("expert_level_balance", 0.01)
    use_aux_loss = config["training"].get("use_aux_loss", False)

    text = open(config["data"]["source_file"], "r").read()
    train_dataset = StoryDataset(text, batch_size, seq_len)

    model = DecoderTransformer(
        ff_function=config["model"]["type"],
        batch_size=batch_size,
        max_len=config["model"]["architecture"]["max_length"],
        vocab_dim=train_dataset.vocab_size,
        emb_dim=config["model"]["architecture"]["embedding_dimension"],
        num_heads=config["model"]["architecture"]["attention_heads"],
        layers=config["model"]["architecture"]["num_layers"],
        ff_dim=config["model"]["architecture"]["feedforward_dimension"],
        shared_experts=config["model"]["architecture"].get("shared_experts", 0),
        num_experts=config["model"]["architecture"].get("num_experts", 0),
        top_k_routers=config["model"]["architecture"].get("top_k_routers", 0),
        routing_type=config["model"]["architecture"].get("routing_type", 0),
        capacity_factor=config["model"]["architecture"].get("capacity_factor", 0),
        pos_embedding_type=config["model"]["architecture"].get("positional_embedding_type", "absolute"),
        attention_type=config["model"]["architecture"].get("attention_type", "MultiHeadAttention")
    )

    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Number of parameters: {num_params}")
    # print(f"Model: {model}")

    print(f"Save file checkpoint: {args.checkpoint}")

    if args.train:
        train(model, train_dataset, lr, epochs, batch_size, seq_len, use_aux_loss, expert_level_balance)

        model.save_weights(args.checkpoint)
    else:
        model.load_weights(args.checkpoint)

    generation = model.generate(
        mx.array([train_dataset.stoi[s] for s in "Would"]).reshape(1, -1),
        config["model"]["architecture"]["max_length"],
        do_sample=True,
        top_k=3,
    )
    print(
        "".join([train_dataset.itos[s.tolist()] for s in list(generation.reshape(-1))])
    )

    print(f"Peak memory usage: {mx.get_peak_memory() / (1024 * 1024)} mb")
