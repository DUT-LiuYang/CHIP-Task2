import numpy as np
import os
import time
import psutil
import csv
import pandas as pd
import random
import sys
# from keras import backend as K
from scipy.stats import pearsonr


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


def geometric_averaging(out_file="results/geometric_fusion_model.csv"):
    results = []

    result_files = [
        "dec_result1_82.29.csv",
        "dec_result2_82.11.csv",  # ++
        "dec_result4_83.04.csv",   # ++
        "dec_result5_83.86.csv",
        "dec_result6_83.01.csv",
        "dec_result7_82.68.csv",
        "dec_result8_83.24.csv",
        "dec_result9_83.14.csv",
        "dec_result10_83.62.csv",
        "dec_result3_84.29.csv",
    ]
    count = len(result_files)

    ids = []

    dir = "results/"
    for index, file in enumerate(result_files):
        csv_reader = csv.reader(open(dir + file, encoding='utf-8'))
        temp = []
        for num, row in enumerate(csv_reader):
            if num == 0:
                continue
            if index == 0:
                ids.append([row[0], row[1]])
            # print(row[2])
            temp.append(float(row[2]))
        results.append(temp[:])
    print(np.shape(results))

    out = open(out_file, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['qid1', 'qid2', 'label'])

    for index, x in enumerate(results[0]):
        temp = 1.0
        # print(str(num))
        for i in range(count):
            # print(str(i) + " " + str(index))
            temp *= results[i][index]
        temp = pow(temp, 1 / count)
        if temp > 0.5:
            csv_write.writerow([ids[index][0], ids[index][1], 1])
        else:
            csv_write.writerow([ids[index][0], ids[index][1], 0])
        # csv_write.writerow([ids[index][0], ids[index][1], temp])
    out.close()


def weighted_vote(out_file="results/vote.csv"):
    results = []
    # result_files = [
    #     # "result4_82.64_0.439.csv",
    #     # "result3_83.94_0.370.csv",
    #     # "result5_84.01_0.385.csv",
    #     # "result2_84.50_0.380.csv",
    #     "result1_85.46_0.358.csv",
    # ]

    result_files = [
        "83.86.csv",    # dec att c
        "83.89.csv",  # dot w --
        "84.02.csv",    # abs sub c magic
        "84.246.csv",   # dot c
        "84.29.csv",    # abs sub w magic
        "84.59.csv",    # abs sub w extra
        "84.595.csv",   # dot w extra
        "84.918.csv",   # abs sub c-> new seed + trainable
        "84.967.csv",   # dot wc
        "85.31.csv",    # abs sub c magic
        # "85.487.csv",      # dot c extra
        "85.51.csv",    # dot c extra
        "85.49.csv",    # abs sub c extra
        "85.55.csv",    # abs sub wc
        "86.023.csv",   # abs sub c
        "86.29.csv",     # dot wc extra
        "86.52.csv",    # abs sub wc extra
    ]

    weights = []
    sum = 0
    for file in result_files:
        temp = file.find('.csv')
        weights.append(float(file[0:temp]))
        sum += weights[-1]

    for index, weight in enumerate(weights):
        weights[index] = weight / sum * 10

    for weight in weights:
        print(weight)

    count = len(result_files)

    ids = []

    dir = "results/"
    for index, file in enumerate(result_files):
        csv_reader = csv.reader(open(dir + file, encoding='utf-8'))
        temp = []
        for num, row in enumerate(csv_reader):
            if num == 0:
                continue
            if index == 0:
                ids.append([row[0], row[1]])
            temp.append(float(row[2]))
        results.append(temp[:])
    print(np.shape(results))

    out = open(out_file, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['qid1', 'qid2', 'label'])

    for index, x in enumerate(results[0]):

        label = 0
        last = -1

        for i in range(count):
            if results[i][index] > 0.5:
                last = 1
                label += weights[i] * 1
            else:

                last = 0
                label += weights[i] * -1

        if label > 0:
            csv_write.writerow([ids[index][0], ids[index][1], 1])
        elif label < 0:
            csv_write.writerow([ids[index][0], ids[index][1], 0])
        else:
            csv_write.writerow([ids[index][0], ids[index][1], last])
    out.close()


def vote(out_file="results/vote.csv"):
    results = []

    result_files = [
        "83.86.csv",    # dec att c++
        "83.89.csv",    # dot w ++
        "84.02.csv",    # abs sub c magic ++
        "84.246.csv",   # dot c++
        "84.29.csv",    # abs sub w magic ++
        "84.59.csv",    # abs sub w extra ++
        "84.595.csv",   # dot w extra ++
        "84.918.csv",   # abs sub c-> new seed + trainable ++
        "84.967.csv",   # dot wc++
        # "85.04.csv",    # DRCN c**+
        # "85.31.csv",    # abs sub c magic --
        "85.39.csv",    # abs sub w extra liu
        # "85.487.csv",      # dot c extra --
        # "85.51.csv",    # dot c extra --
        "85.49.csv",    # abs sub c extra ++
        # "85.55.csv",    # abs sub wc--
        # "86.01.csv",    # abs sub c extra liu --
        "86.023.csv",   # abs sub c
        "86.29.csv",    # dot wc extra ++
        "86.52.csv",    # abs sub wc extra
    ]

    count = len(result_files)

    ids = []

    dir = "results/"
    for index, file in enumerate(result_files):
        csv_reader = csv.reader(open(dir + file, encoding='utf-8'))
        temp = []
        for num, row in enumerate(csv_reader):
            if num == 0:
                continue
            if index == 0:
                ids.append([row[0], row[1]])
            temp.append(float(row[2]))
        results.append(temp[:])
    print(np.shape(results))

    out = open(out_file, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['qid1', 'qid2', 'label'])

    for index, x in enumerate(results[0]):

        p_num = 0
        n_num = 0
        last = -1

        for i in range(count):
            if results[i][index] > 0.5:
                p_num += 1
                last = 1
            else:
                n_num += 1
                last = 0

        if p_num > n_num:
            csv_write.writerow([ids[index][0], ids[index][1], 1])
        elif p_num < n_num:
            csv_write.writerow([ids[index][0], ids[index][1], 0])
        else:
            csv_write.writerow([ids[index][0], ids[index][1], last])
    out.close()


def pearson_value(file1="", file2=""):
    dir = "results/"

    label1 = []
    label2 = []

    csv_reader = csv.reader(open(dir + file1, encoding='utf-8'))

    for num, row in enumerate(csv_reader):
        if num == 0:
            continue
        label1.append(float(row[2]))

    csv_reader = csv.reader(open(dir + file2, encoding='utf-8'))

    for num, row in enumerate(csv_reader):
        if num == 0:
            continue
        label2.append(float(row[2]))

    label1 = np.array(label1)
    label2 = np.array(label2)

    print(pearsonr(label1, label2))

    return pearsonr(label1, label2)


def pearson_value_matrix():
    result_files = [
        "83.89.csv",    # dot w --
        "84.246.csv",   # dot c
        "84.59.csv",    # abs sub w extra
        "84.595.csv",   # dot w extra
        "84.967.csv",   # dot wc
        "85.51.csv",    # dot c extra
        "85.49.csv",    # abs sub c extra
        "85.55.csv",    # abs sub wc
        "86.023.csv",   # abs sub c
        "86.52.csv",    # abs sub wc extra
    ]

    names = [
        "dot w",
        "dot c",
        "abs sub w extra",
        "dot w extra",
        "dot wc",
        "dot c extra",
        "abs sub c extra",
        "abs sub wc",
        "abs sub c",
        "abs sub wc extra",
    ]

    matrix = []

    for index, file1 in enumerate(result_files):
        matrix.append([])
        for i, file2 in enumerate(result_files):
            if index == i:
                matrix[index].append(1)
            else:
                matrix[index].append(pearson_value(file1, file2)[0])

    line = "hello\t"
    for index, file1 in enumerate(names):
        line += names[index] + "\t"

    print(line)

    for index, file1 in enumerate(names):
        line = names[index] + "\t"
        for i, file2 in enumerate(names):
            line += str(matrix[index][i]) + "\t"

        print(line)


def extra_set(file=""):

    dir = "resource/"

    print("read files")
    df_train = pd.read_csv(dir + file)

    q1 = df_train["qid1"].values
    q2 = df_train["qid2"].values
    for i in range(0, q1.shape[0]):
        if q1[i] > q2[i]:
            q1[i], q2[i] = q2[i], q1[i]
    df_train["q1"] = q1
    df_train["q2"] = q2
    print(df_train.head())
    print(df_train.describe())

    q1 = df_train["qid1"].values
    q2 = df_train["qid2"].values
    label = df_train["label"].values

    rows = q1.shape[0]

    dict_1 = dict()

    for i in range(0, rows):
        if label[i] == 1:
            if dict_1.get(q1[i], -1) == -1:
                dict_1[q1[i]] = [q2[i]]
            else:
                dict_1[q1[i]].append(q2[i])

            if dict_1.get(q2[i], -1) == -1:
                dict_1[q2[i]] = [q1[i]]
            else:
                dict_1[q2[i]].append(q1[i])

        if i%5000 == 0:
            sys.stdout.flush()
            sys.stdout.write("#")
    print()
    print(len(dict_1))

    listxy = []
    for x in dict_1:
        listx = dict_1[x]
        if len(listx) > 1:
            listy = listx[:]
            random.shuffle(listy)
            for x,y in zip(listx,listy):
                if x<y:
                    listxy.append([x, y, 1])
                    # listxy.append([1,x,y])
            random.shuffle(listy)
            for x,y in zip(listx,listy):
                if x<y:
                    # listxy.append([1,x,y])
                    listxy.append([x, y, 1])
        if i%5000 == 0:
            sys.stdout.flush()
            sys.stdout.write("#")

    print()
    print(len(listxy))

    random.shuffle(listxy)

    df1 = pd.DataFrame(listxy)
    df1.columns = ["qid1", "qid2", "label"]

    df1.to_csv(dir + "ext_train.csv", index=False)
    print('Complete')


# def broadcast_last_axis(x):
#     """
#     :param x tensor of shape (batch, a, b)
#     :returns broadcasted tensor of shape (batch, a, b, a)
#     """
#     y = K.expand_dims(x, 1) * 0
#     y = K.permute_dimensions(y, (0, 1, 3, 2))
#     return y + K.expand_dims(x)


if __name__ == '__main__':
    vote()
