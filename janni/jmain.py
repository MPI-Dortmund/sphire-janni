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

import argparse
import sys
import json
import os
import h5py

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

DEFAULT_BATCH_SIZE = 4
DEFAULT_PADDING = 24

parser = argparse.ArgumentParser(
    description="Just another noise to noise implementation", add_help=True
)
subparsers = parser.add_subparsers(help="sub-command help")

parser_train = subparsers.add_parser("train", help="train help")
parser_train.add_argument("config_path", help="Path to config.json")
parser_train.add_argument("-g", "--gpu", type=int, default=-1, help="GPU ID to run on")


parser_predict = subparsers.add_parser("predict", help="predict help")
parser_predict.add_argument(
    "input_path", help="Directory / file path with images to denoise"
)
parser_predict.add_argument(
    "output_path", help="Directory / file path to write denoised images"
)
parser_predict.add_argument("model_path", help="File path to trained model")
parser_predict.add_argument("-c", "--config_path", help="File path to config.json")
parser_predict.add_argument("-ol", "--overlap", help="Overlapping in pixels")
parser_predict.add_argument("-bs", "--batch_size", help="Number of patches predicted in parallel")
parser_predict.add_argument(
    "-g", "--gpu", type=int, default=-1, help="GPU ID to run on"
)


def _main_():
    args = parser.parse_args()

    if args.gpu != -1:
        str_gpus = str(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str_gpus

    if "train" in sys.argv[1]:
        config = read_config(args.config_path)

        from . import train

        train.train(
            even_path=config["train"]["even_dir"],
            odd_path=config["train"]["odd_dir"],
            model_out_path=config["train"]["saved_weights_name"],
            movie_path=config["train"]["movie_dir"],
            learning_rate=config["train"]["learning_rate"],
            epochs=config["train"]["nb_epoch"],
            model=config["model"]["architecture"],
            patch_size=(config["model"]["patch_size"], config["model"]["patch_size"]),
            batch_size=config["train"]["batch_size"],
        )

    if "predict" in sys.argv[1]:

        input_path = args.input_path
        output_path = args.output_path
        model_path = args.model_path
        from . import predict
        batch_size = DEFAULT_BATCH_SIZE
        padding = DEFAULT_PADDING

        with h5py.File(model_path, mode="r") as f:
            try:
                import numpy as np
                model = str(np.array((f["model_name"])))
                patch_size = tuple(f["patch_size"])
            except KeyError:
                pass
        if args.config_path is not None:
            config = read_config(args.config_path)
            model = config["model"]["architecture"]
            patch_size = (config["model"]["patch_size"], config["model"]["patch_size"])
            padding = config["model"]["overlap"]
            batch_size = config["train"]["batch_size"]

        if args.overlap is not None:
            padding = int(args.overlap)

        if args.batch_size is not None:
            batch_size = int(args.batch_size)


        predict.predict(
            input_path=input_path,
            output_path=output_path,
            model_path=model_path,
            model= model,
            patch_size=patch_size,
            padding=padding,
            batch_size=batch_size,
        )


def read_config(config_path):
    with open(config_path) as config_buffer:
        try:
            config = json.loads(config_buffer.read())
        except json.JSONDecodeError:
            print(
                "Your configuration file seems to be corruped. Please check if it is valid."
            )
    return config


if __name__ == "__main__":
    _main_()
