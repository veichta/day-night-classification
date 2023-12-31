import argparse
import logging
import os
from typing import Tuple

import numpy as np
import torch
from dataset import Dataset
from model import Classifier
from torch import nn
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory.",
    )
    parser.add_argument(
        "--night_images",
        type=str,
        required=True,
        help="Directory containing night images.",
    )
    parser.add_argument(
        "--day_images",
        type=str,
        required=True,
        help="Directory containing day images.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for data loader.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="Backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Use pretrained backbone.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save model.",
    )
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> Tuple[Dataset, Dataset]:
    """Load data from disk

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        Tuple[Dataset, Dataset]: Train and validation datasets.
    """
    logging.info("Loading dataset...")
    night_data = os.listdir(os.path.join(args.data_dir, args.night_images))
    day_data = os.listdir(os.path.join(args.data_dir, args.day_images))

    night_data = [
        os.path.join(args.data_dir, args.night_images, fname)
        for fname in night_data
        if not os.path.isdir(fname)
    ]
    day_data = [
        os.path.join(args.data_dir, args.day_images, fname)
        for fname in day_data
        if not os.path.isdir(fname)
    ]

    data = night_data + day_data
    labels = [0] * len(night_data) + [1] * len(day_data)

    train_idx = np.random.choice(len(data), int(0.8 * len(data)), replace=False)
    val_idx = np.setdiff1d(np.arange(len(data)), train_idx)

    train_data = [data[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_data = [data[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = Dataset(train_data, train_labels)
    val_dataset = Dataset(val_data, val_labels)

    return train_dataset, val_dataset


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train_dataset, val_dataset = load_data(args)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logging.info("Loading model...")
    model = Classifier(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
    )

    logging.info("Training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # reduce learning rate by 0.1 on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    best_loss = np.inf
    for epoch in range(args.epochs):

        losses = []
        accuracy = []
        model.train()
        model.to(args.device)
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Train {epoch}", ncols=80)
        for batch in train_loader:
            optimizer.zero_grad()

            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accuracy.append((pred.argmax(dim=1) == y).float().mean().item())

            pbar.set_postfix({"loss": np.mean(losses), "accuracy": np.mean(accuracy)})
            pbar.update()

        pbar.close()

        losses = []
        accuracy = []
        model.eval()
        model.to(args.device)
        pbar = tqdm(val_loader, total=len(val_loader), desc=f"Val {epoch}", ncols=80)
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(args.device)
                y = y.to(args.device)

                pred = model(x)
                loss = criterion(pred, y)

                losses.append(loss.item())
                accuracy.append((pred.argmax(dim=1) == y).float().mean().item())

                pbar.set_postfix({"loss": np.mean(losses), "accuracy": np.mean(accuracy)})
                pbar.update()

            pbar.close()

            mean_loss = np.mean(losses)
            scheduler.step(mean_loss)

            if mean_loss < best_loss:
                best_loss = mean_loss
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)

                torch.save(model.state_dict(), os.path.join(args.model_dir, f"{args.backbone}.pt"))

            logging.info(f"Epoch: {epoch} | Loss: {mean_loss}")


if __name__ == "__main__":
    main()
