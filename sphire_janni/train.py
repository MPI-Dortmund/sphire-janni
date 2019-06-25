from . import models
from . import patch_pair_generator as gen
from keras.optimizers import Adam
import os
from . import utils
import mrcfile

def train(even_path,
          odd_path,
          model_out_path,
          movie_path=None,
          learning_rate=0.001,
          epochs=50,
          model="unet",
          patch_size=(1024,1024),
          batch_size=4):

    print("Start training")
    if model == "unet":
        model = models.get_model_unet(input_size=patch_size, kernel_size=(3, 3))
    opt = Adam(lr=learning_rate, epsilon=10 ** -8, amsgrad=True)
    model.compile(optimizer=opt, loss="mse")

    # Read training even/odd micrographs
    even_files = []
    odd_files = []
    for (dirpath, dirnames, filenames) in os.walk(even_path):
        for filename in filenames:
            if filename.endswith(".mrc"):
                even_files.append(os.path.join(dirpath, filename))

    for (dirpath, dirnames, filenames) in os.walk(odd_path):
        for filename in filenames:
            if filename.endswith(".mrc"):
                odd_files.append(os.path.join(dirpath, filename))

    # Create training data for movies which are not in even/odd directory
    try:
        os.makedirs(even_path)
    except FileExistsError:
        pass
    try:
        os.makedirs(odd_path)
    except FileExistsError:
        pass

    filenames_even = list(map(os.path.basename, even_files))
    filenames_odd = list(map(os.path.basename, odd_files))
    for (dirpath, dirnames, filenames) in os.walk(movie_path):
        for filename in filenames:
            if filename.endswith(".mrc"):
                path = os.path.join(dirpath, filename)

                if filename not in filenames_even and filename not in filenames_odd:
                    print("Create even/odd micrograph for:", path)
                    even, odd = utils.create_image_pair(path)

                    out_even_mrc = os.path.join(even_path, filename)
                    with mrcfile.new(out_even_mrc, overwrite=True) as mrc:
                        mrc.set_data(even)

                    even_files.append(out_even_mrc)
                    filenames_even.append(filename)
                    out_odd_mrc = os.path.join(odd_path, filename)
                    with mrcfile.new(out_odd_mrc, overwrite=True) as mrc:
                        mrc.set_data(odd)
                    odd_files.append(out_odd_mrc)
                    filenames_odd.append(filename)

    train_valid_split = int(0.1 * len(even_files))
    train_even_files = even_files[train_valid_split:]
    valid_even_files = even_files[:train_valid_split]
    train_odd_files = odd_files[train_valid_split:]
    valid_odd_files = odd_files[:train_valid_split]
    print(train_even_files)
    print(valid_even_files)
    train_gen = gen.patch_pair_batch_generator(even_images=train_even_files,
                                        odd_images=train_odd_files,
                                        patch_size=patch_size,
                                        batch_size=batch_size,
                                        augment=True)

    valid_gen = gen.patch_pair_batch_generator(even_images=valid_even_files,
                                        odd_images=valid_odd_files,
                                        patch_size=patch_size,
                                        batch_size=batch_size)

    model.fit_generator(generator=train_gen,
                               validation_data=valid_gen,
                               epochs=epochs)

    model.save_weights(model_out_path)
    print("Training done. Weights saved to " + model_out_path)