import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as transforms
from torch.optim import AdamW, Adam, SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
from logging import warn
import argparse
import os
import wandb

from dataset import FER2013, WrapperDataset
from constants import *
from eval import evaluate
from utils import get_device, set_seed

# from eval import evaluate


def train_step(model, images, labels, optimizer, criterion, device="cpu"):
    """
    Takes one train step (forward + backward pass) on a batch

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    images : torch.Tensor
        Batch of images of shape [None, C, H, W]
    labels : torch.Tensor
        Ground truth of shape [None]
    optimizer : torch.optim.Optimizer
        Optimizer algorithm
    criterion : torch.nn.CrossEntropyLoss (or similar)
        Loss Function
    device : str
        Device to use for training

    Returns
    -------
    float
        Loss value from the forward pass
    """
    # Send to device
    images, labels = images.to(device, dtype=torch.float), labels.type(
        torch.LongTensor
    ).to(device)
    model.train()  # Set train mode
    optimizer.zero_grad()  # Reset gradients
    logits = model(images)  # Forward pass
    loss = criterion(logits, labels)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Adjust weights
    return loss


def train(
    model,
    train_dataset,
    val_dataset,
    criterion,
    optimizer,
    epochs,
    batch_size=64,
    device="cpu",
    log_interval=100,
):
    """
    Train the model, log it on W&B, save best model in disk and return the validation metrics

    Parameters
    ----------
    model : torch.nn.Module
    train_dataset : torch.utils.data.Dataset
    val_dataset : torch.utils.data.Dataset
    criterion : torch.nn.CrossEntropyLoss (or similar)
        Loss Function
    optimizer : torch.optim.Optimizer
        Optimizer algorithm like Adam
    epochs : int
        Number of epochs to train
    batch_size : int
        Size of mini-batches
    device : str
        Device to use for training
    log_interval : int, optional
        Number of epochs after which to log to wandb, by default 100

    Returns
    -------
    dict
        Metrics on validation data from best model (based on val AUROC).
        It is a dict containing metrics like "loss", "accuracy", "micro_auroc", "macro_auroc".
    """
    # Initialize data loaders for iterating mini-batches
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Hack: Used to initialize parameters in lazy layers
    _ = model(next(iter(train_loader))[0].to(device, dtype=torch.float))

    # Alert wandb to log this training
    wandb.watch(model, criterion, log="all", log_freq=log_interval)

    best_val_metrics = dict(macro_auroc=0.0)
    batch_num = 0
    for epoch in range(1, epochs + 1):
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch_num += 1
            images, labels = batch_data  # Get data
            # Do forward + backward pass
            train_loss = train_step(
                model, images, labels, optimizer, criterion, device=device
            )

            # Compute metrics for validation data after every few epochs
            if batch_num % log_interval == 0:
                val_metrics = evaluate(
                    model, val_dataset, criterion, batch_size=batch_size, device=device
                )

                # Log in wandb
                log_dict = {
                    "epoch": epoch,
                    "batch_num": batch_num,
                    "train/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/micro_auroc": val_metrics["micro_auroc"],
                    "val/macro_auroc": val_metrics["macro_auroc"],
                }
                # Classwise validation AUROC
                for cls_name in CLASSES:
                    log_dict[f"val/{cls_name}_auroc"] = val_metrics[f"{cls_name}_auroc"]

                # Log metrics in wandb
                wandb.log(log_dict, step=batch_num)

                # Log ROC curve for validation data
                wandb.log(
                    {
                        "val/roc": wandb.plot.roc_curve(
                            val_metrics["ground_truth"],
                            torch.nn.functional.softmax(val_metrics["logits"], dim=-1),
                            labels=CLASSES,
                        )
                    }
                )

                # Update best val auroc
                if val_metrics["macro_auroc"] > best_val_metrics["macro_auroc"]:
                    wandb.run.summary["best_val_micro_auroc"] = val_metrics[
                        "micro_auroc"
                    ]
                    wandb.run.summary["best_val_macro_auroc"] = val_metrics[
                        "macro_auroc"
                    ]
                    wandb.run.summary["best_epoch"] = epoch
                    wandb.run.summary["best_batch_num"] = batch_num
                    for label in CLASSES:
                        wandb.run.summary[f"best_val_{label}_auroc"] = val_metrics[
                            f"{label}_auroc"
                        ]
                    best_val_metrics = val_metrics
                    # Save best model so far in disk
                    torch.save(
                        model.state_dict(), os.path.join(wandb.run.dir, "best_model.pt")
                    )

    # Sync best model to wandb
    wandb.save(os.path.join(wandb.run.dir, "best_model.pt"))

    return best_val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # W&B related parameters
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--project", type=str, default="emotion_recognition")
    parser.add_argument("--entity", type=str, required=True)

    parser.add_argument("--model_name", type=str, required=True)

    # Run-specific parameters
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")

    # Augmentations
    parser.add_argument("--random_zoom", type=float, default=1)
    parser.add_argument("--random_rotation", type=float, default=0)

    # Common hyperparameters
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)

    # Parse the args and remove the non-hyperparameter keys
    run_config = vars(parser.parse_args())
    entity = run_config.pop("entity")
    project = run_config.pop("project")
    model_name = run_config.pop("model_name")
    seed = run_config.pop("seed", None)
    device = get_device(run_config.pop("device"))
    log_interval = run_config.pop("log_interval")

    # Start wandb run
    with wandb.init(
        entity=entity,
        project=project,
        config=run_config,
        group=model_name,
        job_type=None,
    ):
        # Set random seed
        if seed:
            set_seed(seed)

        model = None  # TODO: Insert model object here based on model_name

        # Initialize train dataset with the common transforms
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(IMG_SIZE, antialias=True),
                transforms.ToImage(),
            ]
        )
        dataset = FER2013(root=DEFAULT_DS_ROOT, split="train", transform=transform)

        # 80%-20% Train-validation split
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Define separate transforms for train and val
        train_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.2),
            transforms.RandomResizedCrop(
                IMG_SIZE, scale=(0.8, 1), ratio=(1, 4 / 3), antialias=True
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.2),
            transforms.RandomAffine(degrees=45, translate=(-0.4, 0.4)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToDtype(torch.float, scale=True),
        ]
        val_transform = [
            transforms.ToDtype(torch.float, scale=True),
        ]  # TODO: Maybe weak augmentations here too?

        # Initialize train and validation datasets
        train_dataset = WrapperDataset(train_dataset, transform=train_transform)
        val_dataset = WrapperDataset(val_dataset, transform=val_transform)

        # Initialize optimizer
        optimizer = Adam(model.parameters(), lr=run_config["lr"])

        # Loss function
        criterion = CrossEntropyLoss()  # TODO: Class weight here or upsample in batches

        # Train the model and get the best validation metrics
        best_val_metrics = train(
            model,
            train_dataset,
            val_dataset,
            criterion,
            optimizer,
            run_config["epochs"],
            run_config["batchsize"],
            device,
            log_interval,
        )
