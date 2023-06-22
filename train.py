import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from dataset import Dataset
from model import Classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--night_images", type=str, required=True)
    parser.add_argument("--day_images", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--model_dir", type=str, default="models")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading dataset...")
    night_data = os.listdir(os.path.join(args.data_dir, args.night_images))
    day_data = os.listdir(os.path.join(args.data_dir, args.day_images))

    night_data = [os.path.join(args.data_dir, args.night_images, i) for i in night_data]
    day_data = [os.path.join(args.data_dir, args.day_images, i) for i in day_data]

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

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Train {epoch}")
        losses = []
        accuracy = []
        model.train()
        model.to(args.device)
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

        model.eval()
        model.to(args.device)
        with torch.no_grad():
            losses = []
            accuracy = []
            pbar = tqdm(val_loader, total=len(val_loader), desc=f"Val {epoch}")
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
