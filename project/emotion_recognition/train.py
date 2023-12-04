import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as transforms
from torch.optim import AdamW, Adam, SGD, Adadelta, Adagrad, Adamax, RAdam, NAdam
from torch_optimizer import AdaBelief, AdaBound, Lookahead, Shampoo, Ranger
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from logging import warn
from functools import partial
import argparse
import os
import wandb

from project.emotion_recognition.dataset import (
    COMMON_TRANSFORMS,
    FER2013,
    WrapperDataset,
    get_balanced_sampler,
)
from project.emotion_recognition.constants import *
from project.emotion_recognition.eval import evaluate
from project.emotion_recognition.utils import get_model, set_seed, initialize_weights, EnsembleModel

# from eval import evaluate


def train_step(model, images, labels, optimizer, criterion, device="cuda"):
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
    device="cuda",
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
        It is a dict containing metrics like "loss", "accuracy", "auroc".
    """
    # Initialize data loaders for iterating mini-batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # To upsample minority classes, get weighted shuffled sampler
        sampler=get_balanced_sampler(train_dataset),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Hack: Used to initialize parameters in lazy layers
    _ = model(next(iter(train_loader))[0].to(device, dtype=torch.float))

    # Alert wandb to log this training
    wandb.watch(model, criterion, log="all", log_freq=log_interval)

    best_val_metrics = dict(top1_f1=0.0)
    batch_num = 0
    for epoch in range(1, epochs + 1):
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch_num += 1
            images, labels = batch_data  # Get data
            # Do forward + backward pass
            train_loss = train_step(
                model, images, labels, optimizer, criterion, device=device
            )

            # Log train loss in wandb
            wandb.log({"train/loss": train_loss}, step=batch_num)

            # Compute metrics for validation data after every few epochs
            if batch_num % log_interval == 0:
                val_metrics = evaluate(
                    model, val_dataset, criterion, batch_size=batch_size, device=device
                )

                # Log in wandb
                log_dict = {
                    "epoch": epoch,
                    "batch_num": batch_num,
                    "val/loss": val_metrics["loss"],
                    "val/top1_precision": val_metrics["top1_precision"],
                    "val/top2_precision": val_metrics["top2_precision"],
                    "val/top1_recall": val_metrics["top1_recall"],
                    "val/top2_recall": val_metrics["top2_recall"],
                    "val/top1_f1": val_metrics["top1_f1"],
                    "val/top2_f1": val_metrics["top2_f1"],
                    "val/top1_accuracy": val_metrics["top1_accuracy"],
                    "val/top2_accuracy": val_metrics["top2_accuracy"],
                    "val/auroc": val_metrics["auroc"],
                }
                # Classwise validation AUROC
                for cls_name in CLASSES:
                    log_dict[f"val/{cls_name}_auroc"] = val_metrics[f"{cls_name}_auroc"]

                # Log val metrics in wandb
                wandb.log(log_dict, step=batch_num)

                # Update best val auroc
                if val_metrics["top1_f1"] > best_val_metrics["top1_f1"]:
                    wandb.run.summary["best_val_auroc"] = val_metrics["auroc"]
                    wandb.run.summary["best_top1_precision"] = val_metrics[
                        "top1_precision"
                    ]
                    wandb.run.summary["best_top2_precision"] = val_metrics[
                        "top2_precision"
                    ]
                    wandb.run.summary["best_top1_recall"] = val_metrics["top1_recall"]
                    wandb.run.summary["best_top2_recall"] = val_metrics["top2_recall"]
                    wandb.run.summary["best_top1_f1"] = val_metrics["top1_f1"]
                    wandb.run.summary["best_top2_f1"] = val_metrics["top2_f1"]
                    wandb.run.summary["best_top1_accuracy"] = val_metrics[
                        "top1_accuracy"
                    ]
                    wandb.run.summary["best_top2_accuracy"] = val_metrics[
                        "top2_accuracy"
                    ]
                    wandb.run.summary["best_epoch"] = epoch
                    wandb.run.summary["best_batch_num"] = batch_num
                    for label in CLASSES:
                        wandb.run.summary[f"best_val_{label}_auroc"] = val_metrics[
                            f"{label}_auroc"
                        ]
                    best_val_metrics = val_metrics

                    # Log ROC curve for validation data
                    wandb.log(
                        {
                            "val/roc": wandb.plot.roc_curve(
                                val_metrics["ground_truth"],
                                torch.nn.functional.softmax(
                                    val_metrics["logits"], dim=-1
                                ),
                                labels=CLASSES,
                            )
                        }
                    )

                    # Save best model so far in disk
                    torch.save(
                        model.state_dict(), os.path.join(wandb.run.dir, "best_model.pt")
                    )

    # Sync best model to wandb
    wandb.save(os.path.join(wandb.run.dir, f"best_model.pt"))

    return best_val_metrics, model


def run_experiment(
    run_config, entity, project, train_dataset, val_dataset, criterion,  device="cuda", log_interval=100
):
    # Start wandb run
    with wandb.init(
        entity=entity,
        project=project,
        config=run_config,
        group=model_name + run_config["tag"],
        job_type=None,
    ):
        # Initialize model
        model = get_model(model_name).to(device)

        # Initialize weights
        model.apply(partial(initialize_weights, run_config["init_type"]))

        # Initialize optimizer
        if run_config["optim"] == "adam":
            optimizer = Adam(model.parameters(), lr=run_config["lr"])
        elif run_config["optim"] == "adamw":
            optimizer = AdamW(model.parameters(), lr=run_config["lr"])
        elif run_config["optim"] == "sgd":
            optimizer = SGD(model.parameters(), lr=run_config["lr"])
        elif run_config["optim"] == "adadelta":
            optimizer = Adadelta(model.parameters(), lr=run_config["lr"])
        elif run_config["optim"] == "adagrad":
            optimizer = Adagrad(model.parameters(), lr=run_config["lr"])
        elif run_config["optim"] == "adamax":
            optimizer = Adamax(model.parameters(), lr=run_config["lr"])
        elif run_config["optim"] == "nadam":
            optimizer = NAdam(model.parameters(), lr=run_config["lr"])
        elif run_config["optim"] == "radam":
            optimizer = RAdam(model.parameters(), lr=run_config["lr"])
        elif run_config["optim"] == "adabelief":
            optimizer = AdaBelief(model.parameters())
        elif run_config["optim"] == "adabound":
            optimizer = AdaBound(model.parameters())
        elif run_config["optim"] == "ranger":
            optimizer = Ranger(model.parameters())
        elif run_config["optim"] == "lookahead":
            adam = Adam(model.parameters(), lr=run_config["lr"])
            optimizer = Lookahead(adam, k=5, alpha=0.5)
        elif run_config["optim"] == "shampoo":
            optimizer = Shampoo(model.parameters())
        else:
            raise NotImplementedError

        # Train the model and get the best validation metrics
        best_val_metrics, model = train(
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

    return best_val_metrics, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Location of dataset
    parser.add_argument("--root", type=str, default=DEFAULT_DS_ROOT)

    # W&B related parameters
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--project", type=str, default="emotion-recognition-new")
    parser.add_argument("--entity", type=str, default="deep-learning-ub")
    parser.add_argument("--tag", type=str, default="")

    parser.add_argument(
        "--models",
        nargs="+",
        help="Single model or list of models",
        default=list(MODEL_NAME_MAP.keys()),
    )

    # Run-specific parameters
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")

    # Common hyperparameters
    parser.add_argument(
        "--scale", action="store_true", help="Scale variables before augmentation"
    )
    parser.add_argument(
        "--optim", type=str, choices=
        ["adam", "adamw", "sgd", "adadelta", "adagrad", "adamax", "radam", "nadam", "adabelief", "adabound", "ranger", "lookahead", "shampoo"]
        , default="adam"
    )
    parser.add_argument(
        "--init_type",
        type=str,
        choices=["uniform", "normal", "xavier_uniform", "xavier_normal"],
        default="uniform",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)

    # Parse the args and remove the non-hyperparameter keys
    run_config = vars(parser.parse_args())
    entity = run_config.pop("entity")
    project = run_config.pop("project")
    root = run_config.pop("root")
    seed = run_config.pop("seed", None)
    device = run_config.pop("device")
    log_interval = run_config.pop("log_interval")

    # Set random seed
    if seed:
        set_seed(seed)

    # Initialize train dataset with the common transforms
    dataset = FER2013(root=root, split="train", transform=COMMON_TRANSFORMS)

    # 85%-15% Train-validation split
    train_size = int(len(dataset) * 0.85)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Define separate transforms for train and val
    train_augment = transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomResizedCrop(
                #     IMG_SIZE, scale=(0.9, 1), ratio=(1, 4 / 3), antialias=True
                # ),
                # transforms.RandomAdjustSharpness(sharpness_factor=0.15, p=0.2),
                # transforms.RandomAffine(degrees=45, translate=(0.1, 0.1)),
                # transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
                transforms.ToDtype(torch.float, scale=run_config["scale"]),
            ]
        )
    val_augment = transforms.Compose(
            [
                transforms.ToDtype(torch.float, scale=run_config["scale"]),
            ]
        )  # TODO: Maybe weak augmentations here too?

        # Initialize train and validation datasets
    train_dataset = WrapperDataset(train_dataset, transform=train_augment)
    val_dataset = WrapperDataset(val_dataset, transform=val_augment)

    # Loss function
    criterion = CrossEntropyLoss()

    models = run_config.pop("models")
    trained_models = []
    model_names = []
    for model_name in models:
        print(f"Training model {model_name}")
        val_result, trained_model = run_experiment(
            run_config,
            entity,
            project,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            criterion=criterion,
            device=device,
            log_interval=log_interval,
        )
        trained_models.append(trained_model)
        model_names.append(model_name)
    
    if len(trained_models) > 1:
        group_name = '|'.join(model_names)
        print(f"Evaluating ensemble model {group_name}")
        ensemble_model = EnsembleModel(trained_models)  
        with wandb.init(
        entity=entity,
        project=project,
        config=run_config,
        group=group_name,
        job_type=None,
    ):
            val_metrics = evaluate(
                ensemble_model, val_dataset, criterion, batch_size=run_config["batchsize"], device=device
            )

            # Log in wandb
            log_dict = {
                "epoch": run_config["epochs"],
                "val/loss": val_metrics["loss"],
                "val/top1_precision": val_metrics["top1_precision"],
                "val/top2_precision": val_metrics["top2_precision"],
                "val/top1_recall": val_metrics["top1_recall"],
                "val/top2_recall": val_metrics["top2_recall"],
                "val/top1_f1": val_metrics["top1_f1"],
                "val/top2_f1": val_metrics["top2_f1"],
                "val/top1_accuracy": val_metrics["top1_accuracy"],
                "val/top2_accuracy": val_metrics["top2_accuracy"],
                "val/auroc": val_metrics["auroc"],
            }
            # Classwise validation AUROC
            for cls_name in CLASSES:
                log_dict[f"val/{cls_name}_auroc"] = val_metrics[f"{cls_name}_auroc"]

            # Log val metrics in wandb
            wandb.log(log_dict)

            # Log ROC curve for validation data
            wandb.log(
                {
                    "val/roc": wandb.plot.roc_curve(
                        val_metrics["ground_truth"],
                        torch.nn.functional.softmax(
                            val_metrics["logits"], dim=-1
                        ),
                        labels=CLASSES,
                    )
                }
            )

                # Save best model so far in disk
            torch.save(
                ensemble_model.state_dict(), os.path.join(wandb.run.dir, "best_model.pt")
            )