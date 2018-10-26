import keras
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, BatchNormalization
from All_model.Bi_GRU2_based_Model import LiuModel1
from run import parse_args
import argparse
from utils import PRF, print_metrics
from Models.BaseModel import BaseModel


class LiuModel2(LiuModel1):

    def __init__(self):
        super(LiuModel2, self).__init__()

