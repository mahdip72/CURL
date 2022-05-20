import logging
import torch
import os
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_adobe_5k_data(img_ids_filepath, data_dirpath):
    """ Loads the Samsung image data into a Python dictionary
    :returns: Python two-level dictionary containing the images
    :rtype: Dictionary of dictionaries
    """
    data_dict = dict()

    with open(img_ids_filepath) as f:
        '''
        Load the image ids into a list data structure
        '''
        image_ids = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        image_ids_list = [x.rstrip() for x in image_ids]

    idx = 0
    idx_tmp = 0
    img_id_to_idx_dict = {}

    for root, dirs, files in os.walk(data_dirpath):
        for file in files:
            img_id = file.split("-")[0]

            is_id_in_list = False
            for img_id_test in image_ids_list:
                if img_id_test == img_id:
                    is_id_in_list = True
                    break

            if is_id_in_list:  # check that the image is a member of the appropriate training/test/validation split
                if img_id not in img_id_to_idx_dict.keys():
                    img_id_to_idx_dict[img_id] = idx
                    data_dict[idx] = {}
                    data_dict[idx]['input_img'] = None
                    data_dict[idx]['output_img'] = None
                    idx_tmp = idx
                    idx += 1
                else:
                    idx_tmp = img_id_to_idx_dict[img_id]

                if "input" in root:  # change this to the name of your
                    # input data folder
                    input_img_filepath = file
                    data_dict[idx_tmp]['input_img'] = root + "/" + input_img_filepath

                elif "output" in root:  # change this to the name of your
                    # output data folder
                    output_img_filepath = file
                    data_dict[idx_tmp]['output_img'] = root + "/" + output_img_filepath

    for idx, imgs in data_dict.items():
        assert ('input_img' in imgs)
        assert ('output_img' in imgs)

    return data_dict


class CustomDataset(Dataset):
    def __init__(self, data_dir, target_size, random_resize=False, random_crop=False, test_plot=False):
        img_ids_filepath = ''
        data_dirpath = ''
        self.pairs = load_adobe_5k_data(img_ids_filepath, data_dirpath)
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
