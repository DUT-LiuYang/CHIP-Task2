class Config:
    mode = 'train'          # choices=['train', 'prepare', 'predict', 'evaluate']
    model = 'Toymodel'

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

    need_char_level = True

    batch_size = 64
    epochs = 50
    kfold =0