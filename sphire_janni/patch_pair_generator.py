"""
MIT License

Copyright (c) 2019 Thorsten Wagner

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
        self, even_images, odd_images, patch_size, batch_size=4, augment=False
    ):
        self.even_images = even_images
        self.odd_images = odd_images
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        length = int(np.ceil(len(self.even_images) / self.batch_size))
        return length

    def __getitem__(self, idx):
        # Extracts a random crop from each image
        img_first = idx * self.batch_size
        img_last = img_first + self.batch_size
        if img_last > len(self.even_images):
            img_last = len(self.even_images)

        x = []
        y = []
        for img_index in range(img_first, img_last):

            even_img_path = self.even_images[img_index]
            odd_img_path = self.odd_images[img_index]

            data = utils.read_image(even_img_path)

            start0 = np.random.randint(0, data.shape[0] - self.patch_size[0] + 1)
            end0 = start0 + self.patch_size[0]

            start1 = np.random.randint(0, data.shape[1] - self.patch_size[1] + 1)
            end1 = start1 + self.patch_size[1]
            even = data[start0:end0, start1:end1].astype(np.float32)

            data = utils.read_image(odd_img_path)

            odd = data[start0:end0, start1:end1].astype(np.float32)

            if self.augment:
                flip_selection = np.random.randint(0, 4)
                flip_vertical = flip_selection == 1
                flip_horizontal = flip_selection == 2
                flip_both = flip_selection == 3

                if flip_vertical:
                    even = np.flip(even, 1)
                    odd = np.flip(odd, 1)
                if flip_horizontal:
                    even = np.flip(even, 0)
                    odd = np.flip(odd, 0)
                if flip_both:
                    even = np.flip(np.flip(even, 0), 1)
                    odd = np.flip(np.flip(odd, 0), 1)

                rand_rotation = np.random.randint(4)
                even = np.rot90(even, k=rand_rotation)
                odd = np.rot90(odd, k=rand_rotation)

                if np.random.rand() > 0.5:
                    help = odd
                    odd = even
                    even = help

            even, _, _ = utils.normalize(even)
            odd, _, _ = utils.normalize(odd)
            x.append(even)
            y.append(odd)
        x = np.array(x)
        x = x[:, :, :, np.newaxis]
        y = np.array(y)
        y = y[:, :, :, np.newaxis]

        return x, y

    def on_epoch_end(self):
        index = list(range(len(self.even_images)))
        shuffle(index)
        self.even_images = [self.even_images[i] for i in index]
        self.odd_images = [self.odd_images[i] for i in index]
