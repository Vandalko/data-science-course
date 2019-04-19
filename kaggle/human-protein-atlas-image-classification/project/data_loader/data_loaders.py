from torchvision import datasets, transforms
from base import BaseDataLoader
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T
from imgaug import augmenters as iaa
import pandas as pd
import pathlib


class ProteinDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle, validation_split, num_workers, num_classes, img_size, training=True):
        self.images_df = pd.read_csv(csv_path)
        self.num_classes = num_classes
        self.dataset = ProteinDataset(self.images_df, data_dir, num_classes, img_size, not training, training)
        self.n_samples = len(self.dataset)
        super(ProteinDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        # Dumb stratification.
        validation_split = []
        for idx, (value, count) in enumerate(self.images_df['Target'].value_counts().to_dict().items()):
            if count > 1:
                for _ in range(max(round(split * count), 1)):
                    validation_split.append(value)

        # Oversampling.
        multi = [1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 4]
        validation_split_idx = []
        train_split_idx = []
        for idx, value in enumerate(self.images_df['Target']):
            try:
                validation_split.remove(value)
                validation_split_idx.append(idx)
            except:
                for _ in range(max([multi[int(v)] for v in value.split(' ')])):
                    train_split_idx.append(idx)

        valid_idx = np.array(validation_split_idx)
        train_idx = np.array(train_split_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

class ProteinDataset(Dataset):
    def __init__(self, images_df, base_path, num_classes, img_size, augument=True, training=True):
        base_path = pathlib.Path(base_path)
        self.img_size = img_size
        self.num_classes = num_classes
        self.images_df = images_df.copy()
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.training = training

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        X = self.read_images(index)
        if self.training:
            labels = self.read_labels(index)
            y = np.eye(self.num_classes, dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        X = T.Compose([T.ToPILImage(), T.ToTensor()])(X)
        return X.float(), y

    def read_labels(self, index):
        return np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        images = np.zeros(shape=(self.img_size, self.img_size, 4))
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
