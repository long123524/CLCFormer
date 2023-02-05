import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import Dataset
import glob


class DatasetImageMask(Dataset):

    def __init__(self, file_names):

        self.file_names = file_names

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)
        return img_file_name, image, mask


def load_image(path):

    img = Image.open(path)
    # img = img.resize((224, 224), Image.ANTIALIAS)
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]
    )
    img = data_transforms(img)

    return img


def load_mask(path):

    mask = cv2.imread(path.replace("image", "mask").replace("tif", "tif"), 0)
    mask[mask == 0] = 0
    mask[mask > 0] = 1

    return torch.from_numpy(np.expand_dims(mask, 0)).float()






def get_loader(train_path, batchsize, shuffle=True, num_workers=4, pin_memory=True):

    train_file_names = glob.glob(os.path.join(train_path, "*.tif"))
    dataset = DatasetImageMask(train_file_names)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,drop_last=True)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root):
        print(image_root)
        self.images = load_image(image_root)
        self.gts = load_mask(gt_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt = gt/255.0
        self.index += 1

        return image, gt
