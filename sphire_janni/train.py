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

from . import models
from . import patch_pair_generator as gen
from keras.optimizers import Adam
import os
from . import utils
import mrcfile
import tifffile
from keras.callbacks import ModelCheckpoint

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
    # Read training even/odd micrographs
    even_files, odd_files = calc_even_odd(movie_path, even_path, odd_path,recursive=True)

    trained_model = do_train(even_files,
                             odd_files,
                             model=model,
                             learning_rate=learning_rate,
                             patch_size=patch_size,
                             batch_size=batch_size,
                             epochs=epochs,
                             valid_split = 0.1)
    trained_model.save_weights(model_out_path)
    print("Training done. Weights saved to " + model_out_path)

def calc_even_odd(movie_path,even_path,odd_path, recursive=True):
    '''
    Calculates averages based on the even/odd frames of the movies in movie_path and save the
    respective averages in even_path or odd_path.
    :param movie_path: Path to movie files. Supported are .mrc, .mrcs, .tiff and .tif.
    :param even_path: Path where "even averages" will be written.
    :param odd_path: Path here "odd averages" will be written
    :param recursive: If true, the movie_path is scanned recurively for movies.
    :return:
    '''

    # Read training even/odd micrographs
    even_files = []
    odd_files = []
    for (dirpath, dirnames, filenames) in os.walk(even_path):
        for filename in filenames:
            if filename.endswith(utils.SUPPORTED_FILES):
                even_files.append(os.path.join(dirpath, filename))

    for (dirpath, dirnames, filenames) in os.walk(odd_path):
        for filename in filenames:
            if filename.endswith(utils.SUPPORTED_FILES):
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
            if filename.endswith(utils.SUPPORTED_FILES):
                path = os.path.join(dirpath, filename)

                if filename not in filenames_even and filename not in filenames_odd:
                    print("Create even/odd micrograph for:", path)
                    even, odd = utils.create_image_pair(path)
                    out_even_path = os.path.join(even_path, filename)
                    out_odd_path = os.path.join(odd_path, filename)
                    if path.endswith(("mrcs", "mrc")):
                        with mrcfile.new(out_even_path, overwrite=True) as mrc:
                            mrc.set_data(even)

                        with mrcfile.new(out_odd_path, overwrite=True) as mrc:
                            mrc.set_data(odd)

                    elif path.endswith((".tif", ".tiff")):
                        tifffile.imwrite(out_even_path, even)
                        tifffile.imwrite(out_odd_path, odd)

                    even_files.append(out_even_path)
                    odd_files.append(out_odd_path)
                    filenames_even.append(filename)
                    filenames_odd.append(filename)
        if recursive == False:
            break
    return even_files, odd_files


def do_train(even_files,odd_files, model="unet", learning_rate=0.001, epochs=50, patch_size=(1024,1024), callbacks=[], batch_size=4, valid_split = 0.1):
    '''
    Training noise2noise model.
    :param even_files: List with paths to averages based on the even frames
    :param odd_files: List with paths to averages based on the odd frames
    :param model: Model indentifier. Right now only "unet" is supported.
    :param learning_rate: Learning rate used during training.
    :param epochs: Number of epochs to train the network
    :param patch_size: Patch size in pixel. The network is trained on random patches of the images.
    :param callbacks: Optional callbacks during training. See Keras callbacks for more information.
    :param batch_size: Mini-batch size used during training
    :param valid_split: training-validion split.
    :return: Trained keras model
    '''
    train_valid_split = int(valid_split * len(even_files))
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

    if model == "unet":
        model = models.get_model_unet(input_size=patch_size, kernel_size=(3, 3))
    opt = Adam(lr=learning_rate, epsilon=10 ** -8, amsgrad=True)
    model.compile(optimizer=opt, loss="mse")

    model.fit_generator(generator=train_gen,
                        validation_data=valid_gen,
                        epochs=epochs,
                        callbacks=callbacks)
    return model


