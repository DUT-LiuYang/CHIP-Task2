
import os
import argparse
from All_model import model_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train',
                        choices=['train', 'prepare', 'predict', 'evaluate'])
    parser.add_argument("--model", type=str, default='Toymodel', help='class name of model')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default='RMSprop')
    parser.add_argument('--loss', type=str, default='categorical_crossentropy')

    parser.add_argument('--need_char_level', type=bool, default=True)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--kfold', type=int, default=0)

    return parser.parse_args()


def run(args):
    model = model_dict[args.model](args)
    if args.mode == 'train':
        model.train_model(args.epochs, args.batch_size, args.kfold)
    else:
        model.predict()


if __name__ == '__main__':
    run(parse_args())

