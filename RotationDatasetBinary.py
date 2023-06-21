import albumentations as A
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from itertools import combinations
import cv2
from utils import circ_aug

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(0)
np.random.seed(0)


class Args():
    def __init__(self):
        self.config = 'configs/train_config.yaml'
        self.model_path = None


args = Args()

with open(args.config, 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        quit()


class SeameseDataset(Dataset):
    def __init__(self, label_df, img_size, transform=None):

        self.img_size = img_size
        self.label_df = label_df
        self.path_1_df = label_df['path_1']
        self.path_2_df = label_df['path_2']
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_1_path = self.path_1_df[idx]
        img_2_path = self.path_2_df[idx]

        img_1 = np.asarray(Image.open(img_1_path))
        img_2 = np.asarray(Image.open(img_2_path))

        if self.train:
            # Rotation on random angle
            img_1 = np.array(Image.fromarray(img_1).rotate(np.random.uniform(-int(cfg['random_angle']), int(cfg['random_angle']))))
            img_2 = np.array(Image.fromarray(img_2).rotate(np.random.uniform(-int(cfg['random_angle']), int(cfg['random_angle']))))

            x_shift = np.random.uniform(-20, 20)
            y_shift = np.random.uniform(-20, 20)
            num_rows, num_cols = img_1.shape[:2]
            translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
            img_1 = cv2.warpAffine(img_1, translation_matrix, (num_cols, num_rows))

            x_shift = np.random.uniform(-20, 20)
            y_shift = np.random.uniform(-20, 20)
            num_rows, num_cols = img_2.shape[:2]
            translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
            img_2 = cv2.warpAffine(img_2, translation_matrix, (num_cols, num_rows))

        # Applying augmentations
        if self.transform is not None:
            img_1 = self.transform(image=img_1)["image"]
            img_2 = self.transform(image=img_2)["image"]

        img_1 = np.repeat(img_1[..., np.newaxis], 3, -1)
        img_2 = np.repeat(img_2[..., np.newaxis], 3, -1)

        img_1 = img_1 / np.max(img_1)
        img_1 = np.transpose(img_1, (2, 0, 1))
        img_1 = torch.from_numpy(img_1).type(torch.float32)

        img_2 = img_2 / np.max(img_2)
        img_2 = np.transpose(img_2, (2, 0, 1))
        img_2 = torch.from_numpy(img_2).type(torch.float32)
        return img_1, img_2, torch.tensor(self.target_df[idx])


class DatasetTrain(SeameseDataset):
    def __init__(self, label_path, img_size, transform=None, th=70):
        label_df = pd.read_csv(label_path).reset_index(drop=True)

        label_df = pd.DataFrame(np.array(list(combinations(np.array(label_df[['path', 'bag']]), 2))).reshape((-1, 4)),
                          columns=['path_1', 'bag_1', 'path_2', 'bag_2'])

        label_df['target'] = label_df['bag_1'] == label_df['bag_2']
        label_df = label_df.groupby('target').sample(label_df['target'].sum()).reset_index(drop=True)

        SeameseDataset.__init__(self, label_df, img_size, transform)
        self.target_df = label_df['target'].astype('int')
        self.train = True


class DatasetTest(SeameseDataset):
    def __init__(self, label_path, img_size, transform=None):
        label_df = pd.read_csv(label_path).reset_index(drop=True)
        label_df = pd.DataFrame(np.array(list(combinations(np.array(label_df[['path', 'bag']]), 2))).reshape((-1, 4)),
                                columns=['path_1', 'bag_1', 'path_2', 'bag_2'])
        label_df['target'] = label_df['bag_1'] == label_df['bag_2']
        label_df = label_df.groupby('target').sample(label_df['target'].sum(), random_state=10).reset_index(drop=True)

        SeameseDataset.__init__(self, label_df, img_size, transform)
        self.target_df = label_df['target'].astype('int')
        self.train = False


def data_loader_train(label_path, img_size, batch_size):
    augmentations = [
        A.Resize(width=int(img_size), height=int(img_size)),
        #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=.3, p=0.7),
        A.CoarseDropout(p=0.7, min_width=5, min_height=5, max_width=80, max_height=80, min_holes=2, max_holes=16),
        #A.GaussNoise(p=0.7),
    ]

    transform = A.Compose(augmentations)
    dataset = DatasetTrain(label_path, img_size, transform)
    ds_sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(ds_sampler, batch_size, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
    return data_loader


def data_loader_test(label_path, img_size, batch_size):
    augmentations = [
        A.Resize(width=img_size, height=img_size),
    ]
    transform = A.Compose(augmentations)
    dataset = DatasetTest(label_path, img_size, transform)
    ds_sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(ds_sampler, batch_size, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
    return data_loader

