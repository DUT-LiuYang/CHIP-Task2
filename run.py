
import os
import argparse
from All_model import model_dict
from config import Config


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=Config.mode,
                        choices=['train', 'prepare', 'predict', 'evaluate'])
    parser.add_argument("--model", type=str, default=Config.model, help='class name of model')

    parser.add_argument('--lr', type=float, default=Config.lr)
    parser.add_argument('--dropout', type=float, default=Config.dropout)
    parser.add_argument('--optimizer', type=str, default=Config.optimizer)
    parser.add_argument('--loss', type=str, default=Config.loss)

    parser.add_argument('--need_word_level', type=bool, default=Config.need_word_level)
    parser.add_argument('--need_char_level', type=bool, default=Config.need_char_level)
    parser.add_argument('--word_trainable', type=bool, default=Config.word_trainable)
    parser.add_argument('--char_trainable', type=bool, default=Config.char_trainable)

    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--kfold', type=int, default=Config.kfold)

    parser.add_argument('--save_dir', default="./saved_models/")
    parser.add_argument('--word_emb_dir', default="./instances/word_embed.txt")
    parser.add_argument('--char_emb_dir', default="./instances/char_embed.txt")
    parser.add_argument('--r_dir', default='./resource/')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--selfname', default='zhou')

    return parser.parse_args()


def run(args):
    model = model_dict[args.model](args)
    if args.mode == 'train':
        model.train_model(args.epochs, args.batch_size, args.kfold)
    else:
        model.predict()


if __name__ == '__main__':
    run(parse_args())

