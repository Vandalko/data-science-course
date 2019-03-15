from torchvision import datasets, transforms
from base import BaseDataLoader
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from imgaug import augmenters as iaa
import pandas as pd
import pathlib


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ProteinDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        if training:
            images_df = pd.read_csv(data_dir + '/train.csv')
        else:
            images_df = pd.read_csv(data_dir + '/sample_submission.csv')
        self.dataset = ProteinDataset(images_df, data_dir, not training, training)
        super(ProteinDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ProteinDataset(Dataset):
    def __init__(self, images_df, base_path, augument=True, training=True):
        base_path = pathlib.Path(base_path)
        if training:
            base_path = base_path / "train"
        else:
            base_path = base_path / "test"
        self.images_df = images_df.copy()
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.training = training

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        X = self.read_images(index)
        if self.training:
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
            y = np.eye(28, dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        X = T.Compose([T.ToPILImage(), T.ToTensor()])(X)
        return X.float(), y

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        images = np.zeros(shape=(512, 512, 4))
        r = np.array(Image.open(filename + "_red.png"))
        g = np.array(Image.open(filename + "_green.png"))
        b = np.array(Image.open(filename + "_blue.png"))
        y = np.array(Image.open(filename + "_yellow.png"))
        images[:, :, 0] = r.astype(np.uint8)
        images[:, :, 1] = g.astype(np.uint8)
        images[:, :, 2] = b.astype(np.uint8)
        images[:, :, 3] = y.astype(np.uint8)
        images = images.astype(np.uint8)
        return images

    def augumentor(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),

            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug
