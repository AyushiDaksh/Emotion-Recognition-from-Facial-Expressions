import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.transforms import v2 as transforms
from torchmetrics.functional import auroc as auroc_fn, accuracy as accuracy_fn
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from dataset import FER2013, WrapperDataset
from constants import *
from utils import get_device, get_model

from wandb import Api

api = Api()


@torch.no_grad()
def evaluate(model, dataset, loss_fn, batch_size=64, device="cpu"):
    """
    Evaluate model on given dataset

    Parameters
    ----------
    model : torch.nn.Module
        The given trained model
    dataset : torch.utils.data.Dataset
    loss_fn : torch.nn.CrossEntropyLoss (or similar)
        Loss Function
    batch_size: int
    device : str
        Device on which to run the inference

    Returns
    -------
    dict
        Metrics on the provided data by the given model.
        It is a dict containing metrics like "loss", "accuracy", "macro_auroc".
    """
    model.eval()  # Switch on evaluation model

    # Initialize lists for different metrics
    loss, accuracy, class_auroc, macro_auroc = [], [], [], []
    logits, y = [], []

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Iterate over batches and accumulate metrics
    for batch_X, batch_y in data_loader:
        # Send data to device
        batch_X, batch_y = batch_X.to(device, dtype=torch.float), batch_y.type(
            torch.LongTensor
        )
        logits.append(model(batch_X).cpu())  # Append the logits
        y.append(batch_y)  # Append the predictions

    # Concatenate all results
    logits, y = torch.cat(logits), torch.cat(y)
    loss.append(loss_fn(logits, y))
    accuracy.append(accuracy_fn(logits, y, task="multiclass", num_classes=len(CLASSES)))
    class_auroc.append(
        auroc_fn(logits, y, task="multiclass", num_classes=len(CLASSES), average=None)
    )
    macro_auroc.append(
        auroc_fn(
            logits, y, task="multiclass", num_classes=len(CLASSES), average="macro"
        )
    )

    result = {
        "ground_truth": y,
        "logits": logits,
        "loss": np.mean(loss),
        "accuracy": np.mean(accuracy),
        "macro_auroc": np.mean(macro_auroc),
    }

    # Class-wise AUROC
    class_auroc = class_auroc[0]
    for i, label in enumerate(CLASSES):
        result[f"{label}_auroc"] = class_auroc[i]

    return result


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser()

    # Location of dataset
    parser.add_argument("--root", type=str, default=DEFAULT_DS_ROOT)

    # Wandb-specific params
    parser.add_argument("--runid", type=str, required=True, help="ID of train run")
    parser.add_argument("--project", type=str, default="emotion-recognition")
    parser.add_argument("--entity", type=str, default="deep-learning-ub")

    # Device to run on
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    run_config = parser.parse_args()

    # Start wandb run
    with wandb.init(
        entity=run_config.entity,
        project=run_config.project,
        id=run_config.runid,
        resume="must",
    ) as wandb_r:
        # Get best device on machine
        device = get_device(run_config.device)

        model_name = api.run(wandb_r.path).group
        model = get_model(model_name)

        # Fetch weights from wandb train run
        weights_file = wandb.restore("best_model.pt")
        model.load_state_dict(torch.load(os.path.join(wandb_r.dir, "best_model.pt")))

        # Initialize test dataset with the common transforms
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(IMG_SIZE, antialias=True),
                transforms.ToImage(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToDtype(torch.float, scale=True),
            ]
        )
        dataset = WrapperDataset(
            FER2013(root=run_config.root, split="test", transform=transform),
            transform=test_transform,
        )

        # Loss function
        criterion = CrossEntropyLoss()

        # Evaluate model on test data and get metrics
        metrics = evaluate(
            model,
            dataset,
            criterion,
            batch_size=wandb_r.config.batchsize,
            device=device,
        )

        # Log the summary into W&B
        wandb.run.summary["test/loss"] = metrics["loss"]
        wandb.run.summary["test/accuracy"] = metrics["accuracy"]
        wandb.run.summary["test/macro_auroc"] = metrics["macro_auroc"]
        for cls_name in CLASSES:
            wandb.run.summary[f"test/{cls_name}_auroc"] = metrics[f"{cls_name}_auroc"]

        # Log the ROC plot in W&B
        wandb.log(
            {
                "test/roc": wandb.plot.roc_curve(
                    metrics["ground_truth"],
                    torch.nn.functional.softmax(metrics["logits"], dim=-1),
                    labels=CLASSES,
                )
            }
        )

        # Create confusion matric as a heatmap and log it into W&B and save in results folder on disk
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for idx, cls in enumerate(CLASSES):
            class_truth = (metrics["ground_truth"].numpy() == idx).astype(int)
            class_pred = torch.nn.functional.softmax(metrics["logits"]).numpy()[
                ..., idx
            ]
            fpr[idx], tpr[idx], _ = roc_curve(class_truth, class_pred)
            _ = axes[0].plot(
                fpr[idx],
                tpr[idx],
                label="{} ({:.2f}%)".format(cls, metrics[f"{cls}_auroc"] * 100),
            )
        _ = axes[0].set_title(
            "Test AUROC: {:.2f}%".format(metrics["macro_auroc"] * 100)
        )
        _ = axes[0].legend()

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=metrics["ground_truth"].numpy(),
            y_pred=np.argmax(metrics["logits"], axis=-1),
            display_labels=CLASSES,
            cmap=plt.cm.Blues,
            colorbar=False,
            ax=axes[1],
        )

        fig.tight_layout()

        fig.savefig(f"{model_name}__test_plots.jpg")

        wandb.log({"test/plots": fig})
