class Config:
    mode = 'train'          # choices=['train', 'prepare', 'predict', 'evaluate']
    model = 'CSRA'

    lr = 0.001
    dropout = 0.2

    """
    所有的优化方法
    # sgd = SGD
    # rmsprop = RMSprop
    # adagrad = Adagrad
    # adadelta = Adadelta
    # adam = Adam
    # adamax = Adamax
    # nadam = Nadam
    """
    optimizer = 'RMSprop'
    loss = 'binary_crossentropy'

    need_word_level = True
    need_char_level = False
    word_trainable = False
    char_trainable = False

    batch_size = 64
    epochs = 50
    kfold = 0
