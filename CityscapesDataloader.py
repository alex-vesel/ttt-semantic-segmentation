import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from cityscapeslabels import id2categoryid

IMG_DIR = '/Users/alexvesel/Downloads/leftImg8bit_trainvaltest/leftImg8bit'
FOGGY_IMG_DIR = '/Users/alexvesel/Downloads/leftImg8bit_trainval_foggyDBF/leftImg8bit_foggyDBF'
LABELS_DIR = '/Users/alexvesel/Downloads/gtFine_trainvaltest/gtFine'

BETA_LEVELS = [0.005, 0.01, 0.02]

class CityscapesDataset(Dataset):
    def __init__(self, split, img_size=(256, 256), rotate=True, foggy=False, fog_level=-1):
        self.split = split
        self.foggy = foggy
        self.img_size = img_size
        self.rotate = rotate
        self.paste_car = True

        self.beta = str(BETA_LEVELS[fog_level])
        if foggy:
            self.img_dir = FOGGY_IMG_DIR
        else:
            self.img_dir = IMG_DIR

        self.imgs = []
        self.labels = []

        self._set_files()

        self.mean = np.load('mean.npy')
        self.std = np.load('std.npy')
        self.main_transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((1024, 1024)),
            torchvision.transforms.Resize(img_size),
        ])
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])



    def _set_files(self):
        # get list of folders in the split
        split_path = os.path.join(self.img_dir, self.split)
        folders = os.listdir(split_path)

        # get list of images and labels
        for folder in folders:
            if folder == '.DS_Store':
                continue
            folder_path = os.path.join(split_path, folder)
            imgs = os.listdir(folder_path)
            for img in imgs:
                if self.foggy:
                    if self.beta not in img:
                        continue
                img_path = os.path.join(folder_path, img)
                self.imgs.append(img_path)

                label_path = os.path.join(LABELS_DIR, self.split, folder, img.replace('leftImg8bit', 'gtFine_labelIds').replace(f"_foggy_beta_{self.beta}", ""))
                self.labels.append(label_path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]

        img = Image.open(img).convert('RGB')
        label = Image.open(label)

        #show image
        img = self.main_transform(img)
        # load car front
        if self.paste_car:
            car_front_img = Image.open("car_front.png").convert('RGB')
            car_front_img = self.main_transform(car_front_img)
            # shift car front img to bottom of image
            car_front_img = torchvision.transforms.functional.affine(car_front_img, angle=0, translate=(0, 100), scale=1.4, shear=0)
            car_front_img = torchvision.transforms.functional.adjust_brightness(car_front_img, 0.7)
            # create mask
            mask_arr = np.any((np.array(car_front_img)>0), axis=-1)
            mask = Image.fromarray(255*mask_arr.astype(np.uint8))
            # past car front onto bottom of imge
            img.paste(car_front_img, mask=mask)

        # show image
        # torchvision.transforms.ToPILImage()(img).show()
        img = self.normalize(img)
        label = self.main_transform(label)

        # convert label to one hot
        label = np.array(label)
        if self.paste_car:
            label[mask_arr] = 0
        # convert label to category id
        label = id2categoryid(label)
        # use np one hot
        label = np.eye(8)[label]
        label = label.transpose(2, 0, 1)
        label = torch.tensor(label).float()

        # choose random rotation
        if self.rotate:
            rot_label = np.random.randint(0, 4)
            img = torchvision.transforms.functional.rotate(img, rot_label * 90)
            label = torchvision.transforms.functional.rotate(label, rot_label * 90)
        else:
            rot_label = 0

        return img, label, rot_label


    def sample_rotation(self, img, label):
        rot_label = np.random.randint(0, 4)
        img = torchvision.transforms.functional.rotate(img, rot_label * 90)
        label = torchvision.transforms.functional.rotate(label, rot_label * 90)
        return img, label, rot_label


    def sample_mask(self):
        # sample a few random patches from the image to mask
        mask = np.zeros((256, 256))
        for _ in range(2):
            mask_section = np.random.randint(0, 16)
            mask[mask_section*64:(mask_section+1)*64, mask_section*64:(mask_section+1)*64] = 1
        mask = torch.tensor(mask).float()
        return mask


    def unnormalize(self, img):
        img = img.permute(1, 2, 0)
        img = img.numpy()
        img = img * self.std + self.mean
        img = Image.fromarray((256*img).astype(np.uint8))
        return img

    def calc_normalization(self):
        mean = 0.
        std = 0.
        for item in tqdm(self):
            img, _, _ = item
            img = img.permute(1, 2, 0)
            img = img.numpy()
            mean += img.mean(axis=(0, 1))
            std += img.std(axis=(0, 1))
        mean /= len(self)
        std /= len(self)
        # save mean and std
        np.save('mean2.npy', mean)
        np.save('std2.npy', std)
        return mean, std

if __name__ == '__main__':
    dataset = CityscapesDataset(split='train')
    dataset.calc_normalization()
