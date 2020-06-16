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

'''
from keras.models import Model
from keras.layers import Input, Add, Conv2DTranspose, MaxPooling2D, UpSampling2D, ReLU
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
'''
import tensorflow.keras.layers as layers
import tensorflow.keras as keras


def get_model_unet(input_size=(1024, 1024), kernel_size=(3, 3)):
    inputs = keras.Input(shape=(input_size[0], input_size[1], 1))
    skips = [inputs]

    x = layers.Conv2D(
        name="enc_conv0",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(
        name="enc_conv1",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)  # --- pool_1
    skips.append(x)

    x = layers.Conv2D(
        name="enc_conv2",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)  # --- pool_2
    skips.append(x)

    x = layers.Conv2D(
        name="enc_conv3",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)  # --- pool_3
    skips.append(x)

    x = layers.Conv2D(
        name="enc_conv4",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)  # --- pool_4
    skips.append(x)

    x = layers.Conv2D(
        name="enc_conv5",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)  # --- pool_5 (not re-used)

    x = layers.Conv2D(
        name="enc_conv6",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.UpSampling2D((2, 2))(x)

    x = layers.concatenate([x, skips.pop()])
    x = layers.Conv2D(
        name="dec_conv5",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(
        name="dec_conv5b",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.concatenate([x, skips.pop()])

    x = layers.Conv2D(
        name="dec_conv4",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(
        name="dec_conv4b",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.UpSampling2D((2, 2))(x)

    x = layers.concatenate([x, skips.pop()])

    x = layers.Conv2D(
        name="dec_conv3",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(
        name="dec_conv3b",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.concatenate([x, skips.pop()])
    x = layers.Conv2D(
        name="dec_conv2",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(
        name="dec_conv2b",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([x, skips.pop()])

    x = layers.Conv2D(
        name="dec_conv1a",
        filters=64,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(
        name="dec_conv1b",
        filters=32,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    outputs = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
