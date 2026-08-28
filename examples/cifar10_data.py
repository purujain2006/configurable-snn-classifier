"""A user-supplied dataset, as a worked example.

Point a config at this file and it is used instead of anything built in:

    dataset:
      module: examples/cifar10_data.py
      factory: make_datasets
      root: ./data/cifar10

The only contract is that the function returns a DatasetBundle. The shape
fields describe ONE SAMPLE BEFORE the encoder resizes anything, because the
planner and the feasibility check read them.
"""

import torchvision
import torchvision.transforms as tt

from snnsearch.data.base import DatasetBundle


def make_datasets(root="./data/cifar10", download=True, augment=True):
    # Keep values in [0, 1]: the poisson and temporal encoders read intensity
    # as a probability, so normalizing to zero mean would break them.
    train_tf = [tt.ToTensor()]
    if augment:
        train_tf = [tt.RandomCrop(32, padding=4), tt.RandomHorizontalFlip()] + train_tf

    train = torchvision.datasets.CIFAR10(root=root, train=True, download=download,
                                         transform=tt.Compose(train_tf))
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=download,
                                        transform=tt.Compose([tt.ToTensor()]))

    return DatasetBundle(
        train=train, test=test,
        C=3, H=32, W=32, num_classes=10,
        name="cifar10", is_event=False,
        meta={"augment": augment},
    )
