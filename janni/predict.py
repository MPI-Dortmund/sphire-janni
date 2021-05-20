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

from . import utils
import numpy as np
from . import models
import sys
import os
import mrcfile
import tifffile
import pathlib

def predict(
    input_path,
    output_path,
    model_path,
    model="unet",
    patch_size=(1024, 1024),
    padding=15,
    batch_size=4,
    output_resize_to=None
):

    if model == "unet" or model == b"unet":
        model = models.get_model_unet(input_size=patch_size)
        model.load_weights(model_path)
    else:
        print("Not supported model", model, "Stop")
        sys.exit(0)

    predict_dir(
        input_path=input_path,
        output_path=output_path,
        model=model,
        patch_size=patch_size,
        padding=padding,
        batch_size=batch_size,
        output_resize_to=output_resize_to,
    )

def predict_dir(
    input_path,
    output_path,
    model,
    patch_size=(1024, 1024),
    padding=15,
    batch_size=4,
    output_resize_to=None
):
    """
    Denoises images / movies
    :param input_path: Input path to directory with movies / averages to denoise.
    :param output_path: Folder where results should be written.
    :param model: Trained model
    :param patch_size: Patch size in Pixel. Image will be denoised in patches and then stitched together.
    :param padding: Padding to remove edge effects.
    :param batch_size: Number of patches denoised in parallel.
    :param output_resize_to: The denoised image will be downsized to this dimension.
    :return: List of paths to denoised images
    """
    list_files = []
    for (dirpath, dirnames, filenames) in os.walk(input_path):
        for filename in filenames:
            if filename.endswith(utils.SUPPORTED_FILES):
                path = os.path.join(dirpath, filename)
                list_files.append(path)

    denoise_image_paths = predict_list(
        list_files,
        output_path,
        model,
        patch_size=patch_size,
        padding=padding,
        batch_size=batch_size,
        output_resize_to=output_resize_to
    )

    return denoise_image_paths

def predict_list(
    image_paths,
    output_path,
    model,
    patch_size=(1024, 1024),
    padding=15,
    batch_size=4,
    output_resize_to=None,
    squarify=False,
    fbinning=utils.fourier_binning,
    sliceswise=False
):
    """
    Denoises images / movies
    :param image_paths: List of images to filter
    :param output_path: Folder where results should be written.
    :param model: Trained model
    :param patch_size: Patch size in Pixel. Image will be denoised in patches and then stitched together.
    :param padding: Padding to remove edge effects.
    :param batch_size: Number of patches denoised in parallel.
    :param output_resize_to: The denoised image will be downsized to this dimension. In case of a
    list it should be [height,width]. If a single number if passed the short image side is scaled to
    this size and the long side according the aspect ratio.
    :param squarify: Will make the image square after denoising.
    :param fbinning: Function that is used for binning. By default fourier binning is used
    :param sliceswise: Multi-frame images will be interpreted as movies by default. If you want to
    run the prediction on every slice, set this option to True.
    :return: List of paths to denoised images
    """

    denoise_image_paths = []
    for path in image_paths:
        if path.endswith(utils.SUPPORTED_FILES):
            try:
                os.makedirs(output_path)
            except FileExistsError:
                pass
            pos_path = pathlib.PurePath(path)
            file_outpath = os.path.join(output_path, pos_path.parent.name)
            if not os.path.exists(file_outpath):
                os.makedirs(file_outpath)

            opath = os.path.join(file_outpath, pos_path.name)

            if not os.path.exists(opath):
                if utils.is_movie(path):
                    if not sliceswise:
                        even, odd = utils.create_image_pair(path,fbinning)
                        denoised_even = predict_np(
                            model,
                            even,
                            patch_size=patch_size,
                            padding=padding,
                            batch_size=batch_size,
                        )
                        denoised_odd = predict_np(
                            model,
                            odd,
                            patch_size=patch_size,
                            padding=padding,
                            batch_size=batch_size,
                        )
                        denoised = denoised_even + denoised_odd
                    else:
                        img = utils.read_image(path)
                        img = img.squeeze()
                        denoised = np.zeros(shape=img.shape)
                        for z in range(img.shape[0]):
                            denoised[z] = predict_np(
                                model,
                                img[z,:,:],
                                patch_size=patch_size,
                                padding=padding,
                                batch_size=batch_size,
                            )
                            denoised = denoised.astype(np.float32, copy=False)
                else:
                    img = utils.read_image(path)
                    img = img.squeeze()
                    denoised = predict_np(
                        model,
                        image=img,
                        patch_size=patch_size,
                        padding=padding,
                        batch_size=batch_size,
                    )
                # Write result to disk

                if output_resize_to is not None:
                    if squarify:
                        denoised = squarify(denoised)
                    resize_to = output_resize_to
                    if not isinstance(output_resize_to, (list, tuple)):
                        # Rescale shorter side to output_resize_to
                        ar = denoised.shape[0] / denoised.shape[1]
                        if ar < 1:
                            height = output_resize_to
                            width = int(output_resize_to / ar)
                        else:
                            height = int(output_resize_to * ar)
                            width = output_resize_to
                        resize_to = [height, width]

                    from PIL import Image
                    if len(denoised.shape)==2:
                        denoised = np.array(Image.fromarray(denoised).resize(
                            (resize_to[1], resize_to[0]), resample=Image.BILINEAR))
                    elif len(denoised.shape)==3:
                        resized_denoised = np.zeros(shape=(denoised.shape[0],resize_to[0],resize_to[1]),dtype=np.float32)
                        for z in range(img.shape[0]):
                            resized_denoised[z,:,:] =  np.array(Image.fromarray(denoised[z,:,:]).resize(
                            (resize_to[1], resize_to[0]), resample=Image.BILINEAR))


                print("Write denoised image in", opath)
                if opath.endswith((".mrc", ".mrcs")):
                    with mrcfile.new(opath, overwrite=True) as mrc:
                        mrc.set_data(denoised)
                elif opath.endswith((".tif", ".tiff")):
                    tifffile.imwrite(opath, denoised)
            else:
                print("Already filtered:", path, "Skip!")
            denoise_image_paths.append(opath)

    return denoise_image_paths

def squarify(image, size=None):
    np.random.seed()
    if size is not None:
        target_size=size
    else:
        target_size = np.max(image.shape)
    mean = np.mean(image)
    rectified_image = np.ones(shape=(target_size, target_size)) * mean
    rectified_image[(rectified_image.shape[0]-image.shape[0]):rectified_image.shape[0],0:image.shape[1]] = image

    return rectified_image

def predict_np(model, image, patch_size=(1024, 1024), padding=15, batch_size=4):
    """
    Denoises an image given a keras model.
    :param model: Trained noise2noise model
    :param image: Image as 2D numpy array
    :param patch_size: Patch size in Pixel. Image will be denoised in patches and then stitched together.
    :param padding: Padding to remove edge effects.
    :param batch_size: Number of patches denoised in parallel.
    :return: Denoised image (2D numpy array)
    """
    if image.ndim != 2:
        print("Your image should have only two dimensions. Return none")
        return None
    image, mean, sd = utils.normalize(image)
    img_patches, pads = utils.image_to_patches(
        image, patch_size=patch_size, padding=padding
    )
    denoised_patches = model.predict(
        x=img_patches[:, :, :, np.newaxis], batch_size=batch_size
    )

    denoised_micrograph = utils.patches_to_image(
        denoised_patches, pads, image_shape=image.shape, padding=padding
    )
    denoised_micrograph = denoised_micrograph.astype(np.float32, copy=False)
    denoised_micrograph = denoised_micrograph * sd + mean
    return denoised_micrograph
