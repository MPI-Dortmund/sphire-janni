import argparse
import sys
import json


parser = argparse.ArgumentParser(description="Just another noise to noise implementation",
                                 add_help=True)
subparsers = parser.add_subparsers(help='sub-command help')

parser_train = subparsers.add_parser('train', help='train help')
parser_train.add_argument('config_path', help='Path to config.json')


parser_predict = subparsers.add_parser('predict', help='predict help')
parser_predict.add_argument('input_path', help='Directroy / file path with images to denoise')
parser_predict.add_argument('output_path', help='Directroy / file path to write denoised images')
parser_predict.add_argument('model_path', help='File path to trained model')
parser_predict.add_argument('config_path', help='File path to config.json')

def _main_():
    args = parser.parse_args()

    if "train" in sys.argv[1]:
        config = read_config(args.config_path)

        from . import train
        train.train(even_path=config["train"]["even_dir"],
                    odd_path=config["train"]["odd_dir"],
                    model_out_path=config["train"]["saved_weights_name"],
                    movie_path=config["train"]["movie_dir"],
                    learning_rate=config["train"]["learning_rate"],
                    epochs=config["train"]["nb_epoch"],
                    model=config["model"]["architecture"],
                    patch_size=(config["model"]["patch_size"],
                                config["model"]["patch_size"]),
                    batch_size=config["train"]["batch_size"])

    if "predict" in sys.argv[1]:
        config = read_config(args.config_path)
        input_path = args.input_path
        output_path = args.output_path
        model_path = args.model_path
        from . import predict
        predict.predict_dir(input_path=input_path,
                output_path=output_path ,
                model_path=model_path,
                model=config["model"]["architecture"],
                patch_size=(config["model"]["patch_size"],
                            config["model"]["patch_size"]),
                padding=config["model"]["overlap"],
                batch_size=config["train"]["batch_size"])

def read_config(config_path):
    with open(config_path) as config_buffer:
        try:
            config = json.loads(config_buffer.read())
        except json.JSONDecodeError:
            print("Your configuration file seems to be corruped. Please check if it is valid.")
    return config

if __name__ == "__main__":
    _main_()
