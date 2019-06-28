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

from keras.models import Model
from keras.layers import Input, Add, Conv2DTranspose, MaxPooling2D, UpSampling2D, ReLU
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate


def get_rednet(
    filter_size=(3, 3),
    num_filters=64,
    num_conv_layers=15,
    input_size=(128, 128),
    use_skip=True,
):
    inputs = Input(shape=(input_size[0], input_size[1], 1))

    # Encoder
    skip_conns = []
    x = Conv2D(
        num_filters, filter_size, padding="same", kernel_initializer="he_normal"
    )(inputs)
    x = ReLU()(x)
    for k in range(1, num_conv_layers):
        x = Conv2D(
            num_filters, filter_size, padding="same", kernel_initializer="he_normal"
        )(x)
        x = ReLU()(x)
        if (k + 1) % 2 == 0 and use_skip:
            skip_conns.append(x)

    # Decoder
    for k in range(num_conv_layers - 1):
        x = Conv2DTranspose(
            num_filters, filter_size, padding="same", kernel_initializer="he_normal"
        )(x)
        x = ReLU()(x)
        if (k + 1) % 2 == 0 and use_skip:
            # if k != 13 and k != 11:
            skip = skip_conns.pop()
            x = Add()([x, skip])
            x = ReLU()(x)

    x = Conv2DTranspose(1, filter_size, padding="same", kernel_initializer="he_normal")(
        x
    )
    model = Model(inputs=inputs, outputs=x)
    return model


def get_model_unet(input_size=(1024, 1024), kernel_size=(3, 3)):
    inputs = Input(shape=(input_size[0], input_size[1], 1))
    skips = [inputs]

    x = Conv2D(
        name="enc_conv0",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(
        name="enc_conv1",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)  # --- pool_1
    skips.append(x)

    x = Conv2D(
        name="enc_conv2",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)  # --- pool_2
    skips.append(x)

    x = Conv2D(
        name="enc_conv3",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)  # --- pool_3
    skips.append(x)

    x = Conv2D(
        name="enc_conv4",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)  # --- pool_4
    skips.append(x)

    x = Conv2D(
        name="enc_conv5",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)  # --- pool_5 (not re-used)

    x = Conv2D(
        name="enc_conv6",
        filters=48,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = UpSampling2D((2, 2))(x)

    x = concatenate([x, skips.pop()])
    x = Conv2D(
        name="dec_conv5",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(
        name="dec_conv5b",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    x = concatenate([x, skips.pop()])

    x = Conv2D(
        name="dec_conv4",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(
        name="dec_conv4b",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = UpSampling2D((2, 2))(x)

    x = concatenate([x, skips.pop()])

    x = Conv2D(
        name="dec_conv3",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(
        name="dec_conv3b",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    x = concatenate([x, skips.pop()])
    x = Conv2D(
        name="dec_conv2",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(
        name="dec_conv2b",
        filters=96,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, skips.pop()])

    x = Conv2D(
        name="dec_conv1a",
        filters=64,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(
        name="dec_conv1b",
        filters=32,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = LeakyReLU(alpha=0.1)(x)

    outputs = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
