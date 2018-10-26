from Models.BaseModel import BaseModel
from keras.layers import *

class IM(BaseModel):
    def build_model(self):
        bigru = Bidirectional(GRU(256))