"""
MIT License

Copyright (c) 2019 Max Planck Institute of Molecular Physiology

Author: Thorsten Wagner (thorsten.wagner@mpi-dortmund.mpg.de)
Author: Luca Lusnig (luca.lusnig@mpi-dortmund.mpg.de)
Author: Fabian Schoenfeld (fabian.schoenfeld@mpi-dortmund.mpg.de)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from keras.utils import Sequence
from random import shuffle
import numpy as np
import mrcfile
from . import utils


class patch_pair_batch_generator(Sequence):
    def __init__(
        self, pair_a_images, pair_b_images, patch_size, batch_size=4, augment=False
    ):
        self.pair_a_images = pair_a_images
        self.pair_b_images = pair_b_images
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        length = int(np.ceil(len(self.pair_a_images) / self.batch_size))
        return length

    def __getitem__(self, idx):
        # Extracts a random crop from each image
        img_first = idx * self.batch_size
        img_last = img_first + self.batch_size
        if img_last > len(self.pair_a_images):
            img_last = len(self.pair_a_images)

        x = []
        y = []
        for img_index in range(img_first, img_last):

            pair_a_img_path = self.pair_a_images[img_index]
            pair_b_img_path = self.pair_b_images[img_index]

            data = utils.read_image(pair_a_img_path)

            start0 = np.random.randint(0, data.shape[0] - self.patch_size[0] + 1)
            end0 = start0 + self.patch_size[0]

            start1 = np.random.randint(0, data.shape[1] - self.patch_size[1] + 1)
            end1 = start1 + self.patch_size[1]
            pair_a = data[start0:end0, start1:end1].astype(np.float32)

            data = utils.read_image(pair_b_img_path)

            pair_b = data[start0:end0, start1:end1].astype(np.float32)

            if self.augment:
                flip_selection = np.random.randint(0, 4)
                flip_vertical = flip_selection == 1
                flip_horizontal = flip_selection == 2
                flip_both = flip_selection == 3

                if flip_vertical:
                    pair_a = np.flip(pair_a, 1)
                    pair_b = np.flip(pair_b, 1)
                if flip_horizontal:
                    pair_a = np.flip(pair_a, 0)
                    pair_b = np.flip(pair_b, 0)
                if flip_both:
                    pair_a = np.flip(np.flip(pair_a, 0), 1)
                    pair_b = np.flip(np.flip(pair_b, 0), 1)

                rand_rotation = np.random.randint(4)
                pair_a = np.rot90(pair_a, k=rand_rotation)
                pair_b = np.rot90(pair_b, k=rand_rotation)

                if np.random.rand() > 0.5:
                    help = pair_b
                    pair_b = pair_a
                    pair_a = help

            pair_a, _, _ = utils.normalize(pair_a)
            pair_b, _, _ = utils.normalize(pair_b)
            x.append(pair_a)
            y.append(pair_b)
        x = np.array(x)
        x = x[:, :, :, np.newaxis]
        y = np.array(y)
        y = y[:, :, :, np.newaxis]

        return x, y

    def on_epoch_end(self):
        index = list(range(len(self.pair_a_images)))
        shuffle(index)
        self.pair_a_images = [self.pair_a_images[i] for i in index]
        self.pair_b_images = [self.pair_b_images[i] for i in index]
