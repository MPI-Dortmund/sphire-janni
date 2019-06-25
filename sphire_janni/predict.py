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
                        even = utils.normalize(even)
                        odd = utils.normalize(odd)
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
                    img = utils.normalize(img)
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
    img_patches, pads = utils.image_to_patches(image, patch_size=patch_size, padding=padding)
    denoised_patches = model.predict(x=img_patches[:, :, :, np.newaxis], batch_size=batch_size)

    denoised_micrograph = utils.patches_to_image(denoised_patches, pads,image_shape=image.shape,padding=15)
    denoised_micrograph = denoised_micrograph.astype(np.float32)
    return denoised_micrograph