import logging
import torch
import os
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from readdata import prepare_automate_dataset, prepare_operator1_dataset, prepare_operator2_dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, target_size, random_resize=False, random_crop=False, test_plot=False):
        self.pairs = prepare_operator2_dataset(data_dir)
        self.num_samples = len(self.pairs)
        self.target_size = target_size
        self.random_crop = random_crop
        self.random_resize = random_resize
        self.test_plot = test_plot
        self.crop_transform = torch.nn.Sequential(
            transforms.RandomCrop(size=target_size),
            transforms.RandomHorizontalFlip(p=0.5)
        )

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def normalise_image(img):
        """Normalises image data to be a float between 0 and 1
        """
        img = img.astype('float32') / 255
        return img

    def random_resize_compute(self, input_img, target_img, input_dir):
        if input_img.shape[0] < self.target_size[0] or input_img.shape[1] < self.target_size[1]:
            logging.info(f'####### WARNING #######')
            logging.info(f'size issue check {input_dir}')
            return input_img, target_img

        h_res = random.randint(target_img.shape[1], input_img.shape[1])
        w_res = int(input_img.shape[0] * h_res / input_img.shape[1])

        # if h_res < self.target_size[0] or w_res < self.target_size[1]:
        #     print('WARNING')
        #     return input_img, target_img

        input_img = cv2.resize(input_img, (h_res, w_res))
        target_img = cv2.resize(target_img, (h_res, w_res))
        return input_img, target_img

    def random_crop_compute(self, img, target):
        img = img[None, ...]
        target = target[None, ...]
        img_batch = torch.concat([img, target], dim=0)
        img_batch = self.crop_transform(img_batch)
        return torch.squeeze(img_batch[0]), torch.squeeze(img_batch[1])

    def __getitem__(self, idx):
        input_dir, target_dir = self.pairs[idx]

        input_img = cv2.cvtColor(cv2.imread(input_dir), cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(cv2.imread(target_dir), cv2.COLOR_BGR2RGB)

        # make sure the resolutions to be identical
        target_img = cv2.resize(target_img, (input_img.shape[1], input_img.shape[0]))

        if self.random_resize:
            input_img, target_img = self.random_resize_compute(input_img, target_img, input_dir)

        input_img = self.normalise_image(input_img)
        target_img = self.normalise_image(target_img)

        input_img = torch.tensor(input_img)
        target_img = torch.tensor(target_img)

        input_img = input_img.permute(2, 0, 1)
        target_img = target_img.permute(2, 0, 1)

        if self.random_crop and (
                input_img.shape[1] >= self.target_size[0] and input_img.shape[2] >= self.target_size[1]):
            input_img, target_img = self.random_crop_compute(input_img, target_img)

        if self.test_plot:
            return input_img.permute(1, 2, 0), target_img.permute(1, 2, 0)
        else:
            return input_img, target_img, input_dir


if __name__ == '__main__':
    data_path = 'D:/mehdi/datasets/image enhancement/global enhancing/training datasets/v4/train'

    batch = 8
    training_data = CustomDataset(data_dir=data_path, target_size=(1000, 1000),
                                  random_resize=True, random_crop=True, test_plot=True)
    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=False, num_workers=10)

    for i, j in train_dataloader:
        pass
        # print(i.shape)
        # print(j.shape)
        # for b in range(batch):
        #     cv2.imshow('input', i.numpy()[b][..., ::-1])
        #     cv2.imshow('target', j.numpy()[b][..., ::-1])
        #     cv2.waitKey(0)
