import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.cycle_gan as gan
from trainer import gan_trainer
from utils import Logger


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    class_a_data_loader = get_instance(module_data, 'data_loader_a', config)
    valid_a_data_loader = class_a_data_loader.split_validation()

    class_b_data_loader = get_instance(module_data, 'data_loader_b', config)
    valid_b_data_loader = class_b_data_loader.split_validation()

    # build model architecture
    generator_ab = gan.Generator(input_class_channels=1, output_class_channels=3, internal_channels=64)
    generator_ba = gan.Generator(input_class_channels=3, output_class_channels=1, internal_channels=64)
    discriminator_a = gan.Discriminator(input_class_channels=1, n_output_labels=1, internal_channels=64)
    discriminator_b = gan.Discriminator(input_class_channels=3, n_output_labels=1, internal_channels=64)

    trainer = gan_trainer.GANTrainer(generator_ab, generator_ba, discriminator_a, discriminator_b, class_a_data_loader,
                                     class_b_data_loader, config)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config_file', default=None, type=str,
                           help='config_file file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config_file:
        # load config_file file
        config_file = json.load(open(args.config_file))
        path = os.path.join(config_file['trainer']['save_dir'], config_file['name'])
    elif args.resume:
        # load config_file file from checkpoint, in case new config_file file is not given.
        # Use '--config_file' and '--resume' arguments together to load trained model and train more with changed config_file.
        config_file = torch.load(args.resume)['config_file']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config_file.json', for example.")
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config_file, args.resume)
