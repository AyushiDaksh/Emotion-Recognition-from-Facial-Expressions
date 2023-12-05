import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.transforms import v2 as transforms
from torchmetrics.functional.classification import (
    multiclass_auroc as auroc_fn,
    multiclass_accuracy as accuracy_fn,
    multiclass_precision as precision_fn,
    multiclass_recall as recall_fn,
    multiclass_f1_score as f1_score_fn,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
)
import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from project.emotion_recognition.dataset import (
    COMMON_TRANSFORMS,
    FER2013,
    WrapperDataset,
)
from project.emotion_recognition.constants import *
from project.emotion_recognition.utils import get_model, focal_loss

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
        It is a dict containing metrics like "loss", "accuracy", "auroc".
    """
    model.eval()  # Switch on evaluation model
    torch.set_grad_enabled(False)

    # Initialize lists for different metrics
    loss = []
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
    top1_accuracy = accuracy_fn(logits, y, num_classes=len(CLASSES), top_k=1)
    top2_accuracy = accuracy_fn(logits, y, num_classes=len(CLASSES), top_k=2)
    top1_precision = precision_fn(logits, y, num_classes=len(CLASSES), top_k=1)
    top2_precision = precision_fn(logits, y, num_classes=len(CLASSES), top_k=2)
    top1_recall = recall_fn(logits, y, num_classes=len(CLASSES), top_k=1)
    top2_recall = recall_fn(logits, y, num_classes=len(CLASSES), top_k=2)
    top1_f1 = f1_score_fn(logits, y, num_classes=len(CLASSES), top_k=1)
    top2_f1 = f1_score_fn(logits, y, num_classes=len(CLASSES), top_k=2)
    class_auroc = auroc_fn(logits, y, num_classes=len(CLASSES), average=None)
    auroc = auroc_fn(logits, y, num_classes=len(CLASSES))

    result = {
        "ground_truth": y,
        "logits": logits,
        "loss": np.mean(loss),
        "auroc": auroc,
        "top1_precision": top1_precision,
        "top1_recall": top1_recall,
        "top1_f1": top1_f1,
        "top1_accuracy": top1_accuracy,
        "top2_precision": top2_precision,
        "top2_recall": top2_recall,
        "top2_f1": top2_f1,
        "top2_accuracy": top2_accuracy,
    }

    # Class-wise AUROC
    for i, label in enumerate(CLASSES):
        result[f"{label}_auroc"] = class_auroc[i]

    torch.set_grad_enabled(True)

    return result


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser()

    # Location of dataset
    parser.add_argument("--root", type=str, default=DEFAULT_DS_ROOT)

    # Wandb-specific params
    parser.add_argument("--runid", type=str, required=True, help="ID of train run")
    parser.add_argument("--project", type=str, default="emotion-recognition-new")
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
        device = run_config.device

        model_name = api.run(wandb_r.path).group
        model = get_model(model_name)

        # Fetch weights from wandb train run
        weights_file = wandb.restore("best_model.pt")
        model.load_state_dict(torch.load(os.path.join(wandb_r.dir, "best_model.pt")))

        model = model.to(device)

        # Initialize test dataset with the common transforms
        test_augment = transforms.Compose(
            [
                transforms.ToDtype(torch.float, scale=wandb_r.config["scale"]),
            ]
        )
        dataset = WrapperDataset(
            FER2013(root=run_config.root, split="test", transform=COMMON_TRANSFORMS),
            transform=test_augment,
        )

        # Loss function
        if wandb_r.config["loss"] == "cce":
            criterion = CrossEntropyLoss()
        elif wandb_r.config["loss"] == "focal":
            criterion = focal_loss
        else:
            raise ValueError("Invalid loss function passed")

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
        wandb.run.summary["test/top1_accuracy"] = metrics["top1_accuracy"]
        wandb.run.summary["test/top2_accuracy"] = metrics["top2_accuracy"]
        wandb.run.summary["test/top1_precision"] = metrics["top1_precision"]
        wandb.run.summary["test/top2_precision"] = metrics["top2_precision"]
        wandb.run.summary["test/top1_recall"] = metrics["top1_recall"]
        wandb.run.summary["test/top2_recall"] = metrics["top2_recall"]
        wandb.run.summary["test/top1_f1"] = metrics["top1_f1"]
        wandb.run.summary["test/top2_f1"] = metrics["top2_f1"]
        wandb.run.summary["test/auroc"] = metrics["auroc"]

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
            class_pred = torch.nn.functional.softmax(metrics["logits"], dim=-1).numpy()[
                ..., idx
            ]
            fpr[idx], tpr[idx], _ = roc_curve(class_truth, class_pred)
            _ = axes[0].plot(
                fpr[idx],
                tpr[idx],
                label="{} ({:.2f}%)".format(cls, metrics[f"{cls}_auroc"] * 100),
            )
        _ = axes[0].set_title("Test AUROC: {:.2f}%".format(metrics["auroc"] * 100))
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
