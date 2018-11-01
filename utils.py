import numpy as np
import os
import time
import psutil


def PRF(label: np.ndarray, predict: np.ndarray):
    categories_num = 2
    matrix = np.zeros((categories_num, categories_num), dtype=np.int32)
    label_array = [(label == i).astype(np.int32) for i in range(categories_num)]
    predict_array = [(predict == i).astype(np.int32) for i in range(categories_num)]
    for i in range(categories_num):
        for j in range(categories_num):
            matrix[i, j] = label_array[i][predict_array[j] == 1].sum()

    # (1) confusion matrix
    label_sum = matrix.sum(axis=1, keepdims=True)  # shape: (ca_num, 1)
    matrix = np.concatenate([matrix, label_sum], axis=1)  # or: matrix = np.c_[matrix, label_sum]
    predict_sum = matrix.sum(axis=0, keepdims=True)  # shape: (1, ca_num+1)
    matrix = np.concatenate([matrix, predict_sum], axis=0)  # or: matrix = np.r_[matrix, predict_sum]

    # (2) accuracy
    temp = 0
    for i in range(categories_num):
        temp += matrix[i, i]
    accuracy = temp / matrix[categories_num, categories_num]

    # (3) precision (P), recall (R), and F1-score for each label
    P = np.zeros((categories_num,))
    R = np.zeros((categories_num,))
    F = np.zeros((categories_num,))

    for i in range(categories_num):
        P[i] = matrix[i, i] / matrix[categories_num, i]
        R[i] = matrix[i, i] / matrix[i, categories_num]
        F[i] = 2 * P[i] * R[i] / (P[i] + R[i]) if P[i] + R[i] > 0 else 0

    # # (4) micro-averaged P, R, F1
    # micro_P = micro_R = micro_F = accuracy

    # (5) macro-averaged P, R, F1
    macro_P = P.mean()
    macro_R = R.mean()
    macro_F = 2 * macro_P * macro_R / (macro_P + macro_R) if macro_P + macro_R else 0

    return {'matrix': matrix, 'acc': accuracy,
            'each_prf': [P, R, F], 'macro_prf': [macro_P, macro_R, macro_F]}


def print_metrics(metrics, metrics_type, save_dir=None):
    matrix = metrics['matrix']
    acc = metrics['acc']
    each_prf = [[v * 100 for v in prf] for prf in zip(*metrics['each_prf'])]
    macro_prf = [v * 100 for v in metrics['macro_prf']]
    epoch = metrics['epoch']
    loss = metrics['val_loss']

    lines = ['\n\n**********************************************************************************',
             '*                                                                                *',
             '*                           {}                                  *'.format(
                 time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
             '*                                                                                *',
             '**********************************************************************************\n',
             '------------  Epoch {0}, val_loss: {1}  -----------'.format(epoch, loss),
             'Confusion matrix:',
             '{0:>6}|{1:>6}|{2:>6}|<-- classified as'.format(' ', 'Good', 'Bad'),
             '------|-------------|{0:>6}'.format('-SUM-'),
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('Good', *matrix[0].tolist()),
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('Bad', *matrix[1].tolist()),
             '------|-------------|------',
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('-SUM-', *matrix[2].tolist()),
             '\nAccuracy = {0:6.2f}%\n'.format(acc * 100),
             'Results for the individual labels:',
             '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Good', *each_prf[0]),
             '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Bad', *each_prf[1]),
             '\n<<Official Score>>Macro-averaged result:',
             'P ={0:>6.2f}%, R ={1:>6.2f}%, F ={2:>6.2f}%'.format(*macro_prf),
             '--------------------------------------------------\n']

    [print(line) for line in lines]

    if save_dir is not None:
        with open(os.path.join(save_dir, "{}_logs.log".format(metrics_type)), 'a') as fw:
            [fw.write(line + '\n') for line in lines]


def show_layer_info(layer_name, layer_out):
    print('[layer]: %s\t[shape]: %s \n%s' % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))


def show_memory_use():
    used_memory_percent = psutil.virtual_memory().percent
    strinfo = '{}% memory has been used'.format(used_memory_percent)
    return strinfo
