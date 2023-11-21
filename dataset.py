import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2 as transforms
from torchvision.io import read_image, ImageReadMode
from PIL import Image
import matplotlib.pyplot as plt
import os

from constants import CLASSES


class FER2013(VisionDataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.paths = self._make_dataset_paths()

        if not self.transform:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((48, 48), antialias=True),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )

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
