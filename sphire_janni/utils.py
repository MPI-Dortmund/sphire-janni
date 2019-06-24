import mrcfile
import numpy as np

SUPPORTED_FILES=[".mrc"]

def image_to_patches(image, patch_size=(1024, 1024), padding=15):
    n0 = int(np.ceil((image.shape[0] - 2 * padding) / (patch_size[0] - 2 * padding)))
    n1 = int(np.ceil((image.shape[1] - 2 * padding) / (patch_size[1] - 2 * padding)))
    total = int(n0 * n1)
    patches = np.zeros(shape=(total, patch_size[0], patch_size[1]), dtype=np.float32)
    entry_index = 0

    for off0 in range(n0):
        for off1 in range(n1):

            start0 = off0 * (patch_size[0] - 2 * padding)
            end0 = start0 + (patch_size[0] - 2 * padding)
            start1 = off1 * (patch_size[1] - 2 * padding)
            end1 = start1 + (patch_size[1] - 2 * padding)

            # Add padding
            start0_padded = start0 - padding
            end0_padded = end0 + padding

            if start0_padded < 0:
                # Add the difference to the end to make sure that it is  the same patch size
                end0_padded = end0_padded + np.abs(start0_padded)
                start0_padded = 0
            if end0_padded > image.shape[0]:
                # Add the difference to the start to make sure that it is  the same patch size
                diff = end0_padded - image.shape[0]
                end0_padded = image.shape[0]
                start0_padded = start0_padded - diff

            start1_padded = start1 - padding
            end1_padded = end1 + padding
            if start1_padded < 0:
                # Add the difference to the end to make sure that it is  the same patch size

                end1_padded = end1_padded + np.abs(start1_padded)
                start1_padded = 0

            if end1_padded > image.shape[1]:
                # Add the difference to the start to make sure that it is  the same patch size
                diff = end1_padded - image.shape[1]
                end1_padded = image.shape[1]
                start1_padded = start1_padded - diff

            patches[entry_index] = image[start0_padded:end0_padded, start1_padded:end1_padded]
            entry_index = entry_index + 1

    return patches


def patches_to_image(patches, image_shape=(4096, 4096), padding=15):
    patch_size = (patches.shape[1], patches.shape[2])
    n0 = int(np.ceil((image_shape[0] - 2 * padding) / (patch_size[0] - 2 * padding)))
    n1 = int(np.ceil((image_shape[1] - 2 * padding) / (patch_size[1] - 2 * padding)))

    image = np.zeros(shape=image_shape)
    entry_index = 0
    for off0 in range(n0):
        for off1 in range(n1):
            start0 = off0 * (patch_size[0] - 2 * padding)
            end0 = start0 + (patch_size[0] - 2 * padding)
            pad0_off = 0
            if end0 > image_shape[0]:
                pad0_off = end0 - image_shape[0]
                end0 = image_shape[0]

            start1 = off1 * (patch_size[1] - 2 * padding)
            end1 = start1 + (patch_size[1] - 2 * padding)
            pad1_off = 0
            if end1 > image_shape[1]:
                pad1_off = end1 - image_shape[1]
                end1 = image_shape[1]

            image[start0:end0, start1:end1] = patches[entry_index, (pad0_off + padding):-padding,
                                              (pad1_off + padding):-padding, 0]
            entry_index = entry_index + 1
    return image


def create_image_pair(image_path):
    with mrcfile.open(image_path, permissive=True) as mrc:
        even = np.sum(mrc.data[::2], axis=0).astype(np.float32)
        odd = np.sum(mrc.data[1::2], axis=0).astype(np.float32)
    even = (even-np.mean(even))/np.std(even)
    odd = (odd-np.mean(odd))/np.std(odd)
    return even, odd

def is_movie(path):
    with mrcfile.mmap(path, permissive=True) as mrc:
        return mrc.data.ndim > 2 and mrc.data.shape[0] > 1