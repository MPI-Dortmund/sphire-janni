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
from gooey import Gooey, GooeyParser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

DEFAULT_BATCH_SIZE = 4
DEFAULT_PADDING = 24

ARGPARSER = None

def create_config_parser(parser):
    config_required_group = parser.add_argument_group(
        "Required arguments",
        "The arguments are required to create a config file for JANNI",
    )

    config_required_group.add_argument(
        "config_out_path",
        default="config_janni.json",
        help="Path where you want to write the config file.",
        widget="FileSaver",
        gooey_options={
            "validator": {
                "test": 'user_input.endswith("json")',
                "message": "File has to end with .json!",
            },
            "default_file": "config_janni.json"
        },
    )

    config_required_group.add_argument(
        "--patch_size",
        default=1024,
        type=int,
        help="The image will be denoised in patches. This field defines the patch size..",
    )

    config_required_group.add_argument(
        "--movie_dir",
        help="Path to the directory with the movie files. If an average exists already in even_dir or odd_dir it will be skipped.",
        widget="DirChooser",
    )

    config_required_group.add_argument(
        "--even_dir",
        help="For each movie in movie_dir, an average based on the even frames is calculated and saved in even_dir.",
        widget="DirChooser",
    )

    config_required_group.add_argument(
        "--odd_dir",
        help="For each movie in movie_dir, an average based on the odd frames is calculated and saved in odd_dir.",
        widget="DirChooser",
    )

    config_required_group.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="How many patches are in one mini-batch. If you have memory problems (e.g with cards < 8GB memory), you can try to reduce this value.",
    )

    config_required_group.add_argument(
        "--learning_rate",
        type=float,
        default=10**-3,
        help="Learning rate, should not be changed.",
    )

    config_required_group.add_argument(
        "--nb_epoch",
        type=int,
        default=100,
        help="Number of epochs to train. Default is 100. More epochs seems to only slightly improve the results.",
    )

    config_required_group.add_argument(
        "--saved_weights_name",
        default="janni_model.h5",
        help="Path for saving final weights.",
        widget="FileSaver",
        gooey_options={
            "validator": {
                "test": 'user_input.endswith("h5")',
                "message": "File has to end with .h5!",
            },
            "default_file": "janni_model.h5"
        },
    )



def create_train_parser(parser):
    required_group = parser.add_argument_group(
        "Required arguments", "These options are mandatory to train JANNI"
    )

    required_group.add_argument(
        "config_path",
        help="Path to config.json",
        widget="FileChooser",
        gooey_options={
            "wildcard": "*.json"
        }
    )

    optional_group = parser.add_argument_group(
        "Optional arguments", "These options are optional to train JANNI"
    )

    optional_group.add_argument(
        "-g", "--gpu", type=int, default=-1, help="GPU ID to run on"
    )

def create_predict_parser(parser):
    required_group = parser.add_argument_group(
        "Required arguments", "These options are mandatory to run JANNI"
    )

    required_group.add_argument(
        "input_path",
        help="Directory / file path with images to denoise\n",
        widget="DirChooser",
    )
    required_group.add_argument(
        "output_path",
        help="Directory / file path to write denoised images\n",
        widget="DirChooser",
    )
    required_group.add_argument(
        "model_path",
        help="File path to trained model",
        widget="FileChooser",
        gooey_options={
            "wildcard": "*.h5"
        }
    )

    optional_group = parser.add_argument_group(
        "Optional arguments", "These options are mandatory to run JANNI"
    )
    optional_group.add_argument(
        "-ol",
        "--overlap",
        help="The patches have to overlap to remove artifacts. This is the amount of overlap in pixel.\n",
        default=DEFAULT_PADDING,
    )
    optional_group.add_argument(
        "-bs",
        "--batch_size",
        help="Number of patches predicted in parallel\n",
        default=DEFAULT_BATCH_SIZE,
    )
    optional_group.add_argument(
        "-g", "--gpu", type=int, default=-1, help="GPU ID to run on"
    )

def create_parser(parser):

    subparsers = parser.add_subparsers(help="sub-command help")

    parser_config= subparsers.add_parser("config", help="Create the configuration file for JANNI")
    create_config_parser(parser_config)

    parser_train = subparsers.add_parser("train", help="Train JANNI for your dataset.")
    create_train_parser(parser_train)

    parser_predict = subparsers.add_parser("predict", help="Denoise micrographs using a (pre)trained model.")
    create_predict_parser(parser_predict)



def get_parser():
    parser = GooeyParser(description="Just another noise to noise implementation")
    create_parser(parser)
    return parser


def _main_():
    global ARGPARSER
    import sys

    if len(sys.argv) >= 2:
        if not "--ignore-gooey" in sys.argv:
            sys.argv.append("--ignore-gooey")

    kwargs = {"terminal_font_family": "monospace", "richtext_controls": True}
    Gooey(
        main,
        program_name="JANNI",
        #image_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../icons"),
        progress_regex=r"^.* \( Progress:\s+(-?\d+) % \)$",
        disable_progress_bar_animation=True,
        tabbed_groups=True,
        **kwargs
    )()


def main(args=None):

    if args is None:
        parser = get_parser()
        args = parser.parse_args()



    if "config" in sys.argv[1]:
        generate_config_file(config_out_path=args.config_out_path,
                             architecture="unet",
                             patch_size=args.patch_size,
                             movie_dir=args.movie_dir,
                             even_dir=args.even_dir,
                             odd_dir=args.odd_dir,
                             batch_size=args.batch_size,
                             learning_rate=args.learning_rate,
                             nb_epoch=args.nb_epoch,
                             saved_weights_name=args.saved_weights_name)
    else:
        if isinstance(args.gpu, list):
            if len(args.gpu) == 1:
                if args.gpu[0] != "-1":
                    str_gpus = args.gpu[0].strip().split(" ")
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_gpus)
        elif args.gpu != -1:
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

        elif "predict" in sys.argv[1]:

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
                    print("Error on loading model", model_path)
                    sys.exit(0)

            if args.overlap is not None:
                padding = int(args.overlap)

            if args.batch_size is not None:
                batch_size = int(args.batch_size)

            predict.predict(
                input_path=input_path,
                output_path=output_path,
                model_path=model_path,
                model=model,
                patch_size=patch_size,
                padding=padding,
                batch_size=batch_size,
            )

def generate_config_file(config_out_path,
                         architecture,
                         patch_size,
                         movie_dir,
                         even_dir,
                         odd_dir,
                         batch_size,
                         learning_rate,
                         nb_epoch,
                         saved_weights_name):
    model_dict = {'architecture': architecture,
                  'patch_size': patch_size,
                  }

    train_dict = {'movie_dir': movie_dir,
                  'even_dir': even_dir,
                  'odd_dir': odd_dir,
                  'batch_size': batch_size,
                  'learning_rate': learning_rate,
                  'nb_epoch': nb_epoch,
                  "saved_weights_name": saved_weights_name,
                  }

    from json import dump
    dict = {"model": model_dict, "train": train_dict}
    with open(config_out_path, 'w') as f:
        dump(dict, f, ensure_ascii=False, indent=4)
    print("Wrote config to", config_out_path)

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
