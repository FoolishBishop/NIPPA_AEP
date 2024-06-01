from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from numpy.core.multiarray import array as array
from torch import nn, Tensor
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from lightorch.nn.dnn import DeepNeuralNetwork
from typing import Sequence, List, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
from lightorch.training.supervised import Module
from lightorch.nn.criterions import LighTorchLoss
import pandas as pd
import numpy as np
from lightning.pytorch import LightningDataModule

# R2 criterion


class Pearson(LighTorchLoss):
    def __init__(self, factor: float) -> None:
        super().__init__([self.__class__.__name__], {self.__class__.__name__: factor})

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Compute means
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        # Compute deviations from means
        xm = x - mean_x
        ym = y - mean_y

        # Compute the correlation
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm**2)) * torch.sqrt(torch.sum(ym**2))
        r = r_num / r_den

        return r, r * self.factors[self.__class__.__name__]


# Feature extractor


class FeatureExtractorVGG19(nn.Module):
    def __init__(self, architecture: Sequence[int], activations: Sequence[int]):
        super(FeatureExtractorVGG19, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights

        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = DeepNeuralNetwork(1280, architecture, activations)

        self.transform = tt.Compose(
            [tt.ToTensor(), VGG19_Weights.IMAGENET1K_V1.transforms(antialias=True)]
        )

    def forward(self, x):
        out = self.model(x)
        return out


class ContrastiveLearning(Module):
    def __init__(self, model1: nn.Module, **hparams) -> None:
        super().__init__(**hparams)

        self.model1 = model1

        self.model2 = FeatureExtractorVGG19(self.architecture, self.activations)


class NonLinearConstrastiveLearning(ContrastiveLearning):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        architecture: Sequence[int],
        activations: Sequence[int],
    ) -> None:
        super().__init__(
            nn.LSTM(
                7,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            ),
            architecture,
            activations,
        )

    def loss_forward(self, batch: Tensor, idx: Tensor) -> Tensor:
        x, X = batch

        pred1 = self.model1(x)
        pred2 = self.model2(X)

        return dict(input=pred1, targeet=pred2)


class LinearConstrastiveLearning(ContrastiveLearning):
    def __init__(self, architecture: Sequence[int], activations: Sequence[int]) -> None:
        super().__init__(nn.LSTM(7, 100), architecture, activations)

    def loss_forward(self, batch: Tensor, idx: Tensor) -> Tensor:
        x, X = batch

        pred1 = self.model1(x)
        pred2 = self.model2(X)

        return dict(input=pred1, targeet=pred2)


class CosmicRayDataset(Dataset):
    def __init__(self, tabular: pd.DataFrame, images: List[np.array]) -> None:
        self.tabular = torch.from_numpy(tabular.values)
        self.images = [torch.from_numpy(img) for img in images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self.tabular[index, :], self.images[index]  # Already shifted


class CosmicModule(LightningDataModule):
    def __init__(
        self,
        dataset: CosmicRayDataset,
        batch_size: int,
        pin_memory: bool = True,
        num_workers: int = 12,
    ) -> None:
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.dataset = dataset

    def setup(self, stage: str) -> None:
        train_len = round(0.8 * self.dataset)
        val_len = len(self.dataset) - train_len
        self.train_ds, self.val_ds = random_split(self.dataset, [train_len, val_len])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            self.batch_size,
            True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            self.batch_size * 2,
            False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


def get_unsup_dataset(images: List[np.array], model) -> np.array:
    out: Tensor = Tensor([torch.from_numpy(img) for img in images])
    out: Tensor = model(out)
    out: np.array = out.squeeze(0).numpy()
    return out
