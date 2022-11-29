import argparse
import importlib
from utils.utils import *
import yaml
import logging
MODEL_DIR=None
DATA_DIR = 'YOUR DATA DIR'
PROJECT='stdu'

def dict2namespace(dicts):
    for i in dicts:
        if isinstance(dicts[i], dict):
            dicts[i] = dict2namespace(dicts[i]) 
    ns = argparse.Namespace(**dicts)
    return ns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='librispeech',
                        choices=['FMC', 'nsynth-100', 'nsynth-200', 'nsynth-300', 'nsynth-400', 'librispeech',
                        'f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-save_path', type=str, default='')
    parser.add_argument('-config', type=str, default="configs/default.yaml") 
    parser.add_argument('-debug', action='store_true')
    # about training
    parser.add_argument('-gpu', default='0')
    args = parser.parse_args()
    with open(args.config, 'r') as config:
        cfg = yaml.safe_load(config) 
    cfg = cfg['train']
    cfg.update(vars(args))
    # args = argparse.Namespace(**cfg)
    args = dict2namespace(cfg)
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()