from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import ImageOps
from PIL import Image
import copy


class ScrapeImageDataset(Dataset):
    """Scrape Image Dataset class"""

    def __init__(self, root_dir, img_size, transform=None):
        self.img_list = os.listdir(root_dir)
        self.root_dir = root_dir  # root directory holding actual image files
        self.transform = transform  # instance of torch transform class (need to double check this)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.img_list[index])
        im = Image.open(img_path)
        new_im = ImageOps.fit(im, self.img_size)

        trans = transforms.ToTensor()
        img = trans(new_im)

        if self.transform:
            img = self.transform(img)

        return img

    def remove_outliers(self, reconstructed_imgs):
        cosine_sim_list = []
        for i, recon_img in enumerate(reconstructed_imgs):
            original_img = self.__getitem__(i)
            cosine_sim = torch.cosine_similarity(original_img, recon_img)
            cosine_sim_product = 1
            for mag in cosine_sim:
                cosine_sim_product *= mag
            cosine_sim_list.append(cosine_sim_product)
        cosine_sim_tensor = torch.from_numpy(cosine_sim_list)
        per_25 = torch.kthvalue(cosine_sim_tensor, cosine_sim_tensor.shape[0] * 1 // 4)[0]
        per_75 = torch.kthvalue(cosine_sim_tensor, cosine_sim_tensor.shape[0] * 3 // 4)[0]
        iqr = per_75 - per_25
        bottom_range = per_25 - (1.5 * iqr)
        top_range = per_75 + (1.5 * iqr)

        outlier_indices = []
        for i, v in enumerate(cosine_sim_tensor):
            if v > top_range or v < bottom_range:
                outlier_indices.append(i)

        for i, v in enumerate(self.img_list):
            if i in outlier_indices:
                os.remove(os.path.join(self.root_dir, v))

    def generate_new_images(self):
        pass


class ImageTransformation(object):
    def __int__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size


class Resize(ImageTransformation):
    """Resizes an image to a specified size


    """
    def __init__(self, output_size):
        ImageTransformation.__int__(self, output_size)

    def __call__(self, image):

        img = transform.resize(image, self.output_size)

        return img


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
