'''
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
'''

import mrcfile
import tifffile
import numpy as np
from . import utils
SUPPORTED_FILES=(".mrc",".mrcs",".tiff",".tif")

def image_to_patches(image, patch_size=(1024, 1024), padding=15):
    roi_size = (patch_size[0] - 2 * padding, patch_size[1] - 2 * padding)

    pad_before0 = padding
    diff0 = (image.shape[0] - roi_size[0] * (image.shape[0] // roi_size[0]))
    if diff0 == 0:
        pad_after0 = 0
    else:
        pad_after0 = patch_size[0] - diff0

    pad_before1 = padding
    diff1 = (image.shape[1] - roi_size[1] * (image.shape[1] // roi_size[1]))
    if diff1 == 0:
        pad_after1 = 0
    else:
        pad_after1 = patch_size[1] - (
                    image.shape[1] - roi_size[1] * (image.shape[1] // roi_size[1]))
    pads = [(pad_before0, pad_after0), (pad_before1, pad_after1)]

    n0 = int(np.ceil(image.shape[0] / roi_size[0]))
    n1 = int(np.ceil(image.shape[1] / roi_size[0]))

    image = np.pad(image, pads, mode="symmetric")

    total = int(n0 * n1)
    patches = np.zeros(shape=(total, patch_size[0], patch_size[1]), dtype=np.float32)

    entry_index = 0
    for off0 in range(n0):
        for off1 in range(n1):
            start0 = off0 * roi_size[0]
            end0 = start0 + patch_size[0]
            start1 = off1 * roi_size[1]
            end1 = start1 + patch_size[1]
            patches[entry_index] = image[start0:end0, start1:end1]
            entry_index = entry_index + 1

    return patches, pads


def patches_to_image(patches, pads, image_shape=(4096, 4096), padding=15):
    patch_size = (patches.shape[1], patches.shape[2])

    roi_size = (patch_size[0] - 2 * padding, patch_size[1] - 2 * padding)

    entry_index = 0
    image = np.zeros(
        shape=(image_shape[0] + pads[0][0] + pads[0][1], image_shape[1] + pads[1][0] + pads[1][1]))
    n0 = int(np.ceil(image_shape[0] / roi_size[0]))
    n1 = int(np.ceil(image_shape[1] / roi_size[0]))
    for off0 in range(n0):
        for off1 in range(n1):

            start0 = pads[0][0] + off0 * roi_size[0]
            end0 = start0 + roi_size[0]
            start1 = pads[1][0] + off1 * roi_size[1]
            end1 = start1 + roi_size[1]
            if off0 == 0 and off1 > 0 and off1 < (n1 - 1):
                image[0:end0, start1:end1] = patches[entry_index, 0:-padding, padding:-padding, 0]
            elif off0 > 0 and off0 < (n0 - 1) and off1 == 0:
                image[start0:end0, 0:end1] = patches[entry_index, padding:-padding, 0:-padding, 0]
            elif off0 == 0 and off1 == 0:
                image[0:end0, 0:end1] = patches[entry_index, 0:-padding, 0:-padding, 0]
            elif off0 > 0 and off0 < (n0 - 1) and off1 == (n1 - 1):
                roi = patches[entry_index, padding:-padding, padding:, 0]
                image[start0:end0, start1:(start1 + roi.shape[1])] = roi
            elif off0 == (n0 - 1) and off1 > 0 and off1 < (n1 - 1):
                roi = patches[entry_index, padding:, padding:-padding, 0]
                image[start0:(start0 + roi.shape[0]), start1:end1] = roi
            elif off0 == (n0 - 1) and off1 == (n1 - 1):
                roi = patches[entry_index, padding:, padding:, 0]
                image[start0:(start0 + roi.shape[0]), start1:(start1 + roi.shape[1])] = roi
            elif off0 == 0 and off1 == (n1 - 1):
                roi = patches[entry_index, :-padding, padding:, 0]
                image[:end0, start1:(start1 + roi.shape[1])] = roi
            elif off0 == (n0 - 1) and off1 == 0:
                roi = patches[entry_index, padding:, :-padding, 0]
                image[start0:(start0 + roi.shape[0]), :end1] = roi
            else:
                image[start0:end0, start1:end1] = patches[entry_index, padding:-padding,
                                                  padding:-padding, 0]
            entry_index = entry_index + 1
    image = image[pads[0][0]:-pads[0][1], pads[1][0]:-pads[1][1]]
    return image


def create_image_pair(image_path):

    data = utils.read_image(image_path)
    even = np.sum(data[::2], axis=0).astype(np.float32)
    odd = np.sum(data[1::2], axis=0).astype(np.float32)

    return even, odd

def normalize(img):
    mean = np.mean(img)
    sd = np.std(img)
    img = (img - mean) / sd
    img[img < -3] = -3
    img[img > 3] = 3
    return img, mean, sd

def read_image(path):
    if path.endswith((".tif",".tiff")):
        return tifffile.imread(path)
    elif path.endswith(("mrc","mrcs")):
        with mrcfile.mmap(path, permissive=True) as mrc:
            return mrc.data
    else:
        print("Image format not supported. File: ", path)
        return None

def is_movie(path):
    if path.endswith((".tif",".tiff")):
        tif = tifffile.TiffFile(path)
        return len(tif.pages)>1
    elif path.endswith(("mrc","mrcs")):
        with mrcfile.mmap(path, permissive=True) as mrc:
            return mrc.data.ndim > 2 and mrc.data.shape[0] > 1