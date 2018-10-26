from Models.BaseModel import BaseModel
from keras.layers import *

class IM(BaseModel):
    def build_model(self):
        dropout = self.args.dropout
        bigru = Bidirectional(GRU(256, dropout=dropout))
