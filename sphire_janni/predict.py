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

from . import utils
import numpy as np
from . import models
import sys
import os
import mrcfile

def predict_dir(input_path,
                output_path ,
                model_path,
                model="unet",
                patch_size=(1024,1024),
                padding=15,
                batch_size=4):
    if model=="unet":
        model = models.get_model_unet(input_size=patch_size)
        model.load_weights(model_path)
    else:
        print("Not supported model",model,"Stop")
        sys.exit(0)
    for (dirpath, dirnames, filenames) in os.walk(input_path):
        for filename in filenames:
            if filename.endswith(".mrc"):
                path = os.path.join(dirpath, filename)
                if utils.is_movie(path):
                        even, odd = utils.create_image_pair(path)
                        denoised_even = predict_np(model,
                                   even,
                                   patch_size=patch_size,
                                   padding=padding,
                                   batch_size=batch_size)
                        denoised_odd = predict_np(model,
                                                  odd,
                                                   patch_size=patch_size,
                                                   padding=padding,
                                                   batch_size=batch_size)
                        denoised = (denoised_even+denoised_odd)/2
                else:
                    with mrcfile.open(path, permissive=True) as mrc:
                        img = mrc.data
                    img = img.squeeze()
                    denoised = predict_np(model,
                                               image=img,
                                               patch_size=patch_size,
                                               padding=padding,
                                               batch_size=batch_size)
                # Write result to disk
                try:
                    os.makedirs(output_path)
                except FileExistsError:
                    pass
                opath = os.path.join(output_path, filename)
                print("Write denoised image", opath)
                with mrcfile.new(opath, overwrite=True) as mrc:
                    mrc.set_data(denoised)



def predict_np(model, image, patch_size=(1024, 1024), padding=15,batch_size=4):
    image, mean, sd = utils.normalize(image)
    img_patches, pads = utils.image_to_patches(image, patch_size=patch_size, padding=padding)
    denoised_patches = model.predict(x=img_patches[:, :, :, np.newaxis], batch_size=batch_size)

    denoised_micrograph = utils.patches_to_image(denoised_patches, pads,image_shape=image.shape,padding=padding)
    denoised_micrograph = denoised_micrograph.astype(np.float32)
    denoised_micrograph = (denoised_micrograph+mean)*sd
    return denoised_micrograph