from torchvision.datasets import VisionDataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
import os

from project.emotion_recognition.constants import CLASSES, IMG_SIZE

COMMON_TRANSFORMS = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE, antialias=True),
        transforms.ToImage(),
    ]
)


class FER2013(VisionDataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.paths = self._make_dataset_paths()
        self.transform = transform

    def _make_dataset_paths(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"{self.root} folder doesn't exist")

        paths = []
        for class_idx, target_class in enumerate(sorted(CLASSES)):
            class_dir = os.path.join(self.root, self.split, target_class)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"{class_dir} not found in dataset folder")
            for img_file in os.scandir(class_dir):
                paths.append((img_file.path, class_idx))

        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path, target = self.paths[index]
        image = read_image(path, mode=ImageReadMode.GRAY)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def display(self, index):
        path, target = self.paths[index]
        image = Image.open(path)
        plt.imshow(image, cmap="gray")
        plt.title(CLASSES[target])
        plt.axis("off")
        plt.show()


class WrapperDataset(Dataset):
    """
    Dataset to wrap subsets of datasets and their transformations
    This is needed to have different transformations for train and val subsets

    Parameters
    ----------
    subset : torch.utils.data.Dataset
        The dataset to use
    transform : Function
        The transformation pipeline. If None, no transformations are applied
    """

    def __init__(self, subset, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def display(self, index):
        self.subset.dataset.display(self.subset.indices[index])


def get_balanced_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Get a shuffled sampler where the probability of picking a class sample is
    inversely proportional to its frequency

    Parameters
    ----------
    dataset : torch.utils.data.Dataset

    Returns
    -------
    WeightedRandomSampler
    """
    num_classes = len(CLASSES)
    # Count how many times each class appears in the dataset
    class_counts = [0] * num_classes
    for _, label in dataset:
        class_counts[label] += 1

    # Create weight for each class based on its frequency
    weight_per_class = [
        len(dataset) / (num_classes * class_counts[i]) for i in range(num_classes)
    ]

    # Assign a weight to each sample in the dataset
    weights = [weight_per_class[label] for _, label in dataset]

    # Create a WeightedRandomSampler object with these weights
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    return sampler
