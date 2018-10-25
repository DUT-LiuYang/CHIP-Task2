import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold

from preprocess.csv_reader import CsvReader


class ExampleReader:

    def __init__(self):
        self.dir = "../resource/"

    def get_embedding_matrix(self, name=""):
        embedding_matrix_file = self.dir + name
        embedding_matrix = []
        rf = open(embedding_matrix_file, 'r')
        while True:
            line = rf.readline()
            if line == "":
                break
            embedding_matrix.append([float(x) for x in line.split()])
        rf.close()
        return np.array(embedding_matrix)

    def question_pairs2question_inputs(self, inputs=[], id_questions={}):
        question_inputs1 = []
        question_inputs2 = []
        max_len = 0
        for q1, q2 in inputs:
            temp = id_questions[q1]
            question_inputs1.append(temp[:])
            max_len = max([max_len, len(temp)])
            temp = id_questions[q2]
            question_inputs2.append(temp[:])
            max_len = max([max_len, len(temp)])
        print("max length of question is " + str(max_len))
        question_inputs1 = pad_sequences(question_inputs1, maxlen=max_len, padding='post')
        question_inputs2 = pad_sequences(question_inputs2, maxlen=max_len, padding='post')
        return np.array(question_inputs1), np.array(question_inputs2)

    def get_Indicator_matrix(self, q1=np.array([]), q2=np.array([]), max_len=39):
        res = []
        count = 0
        for (s1, s2) in zip(q1, q2):
            temp = []
            for index, w1 in enumerate(s1):
                if w1 == 0:
                    temp.append([0] * max_len)
                    continue
                else:
                    temp.append([])
                for w2 in s2:
                    if w2 == w1:
                        temp[index].append(1)
                        # print(str(w1) + " " + str(w2))
                    else:
                        temp[index].append(0)
            res.append(temp[:][:])
            count += 1
        return np.array(res, dtype='float32')


if __name__ == '__main__':
    csv_reader = CsvReader()
    print("read data from train.csv...")
    train_data, train_label = csv_reader.read_csv(name="train.csv", train=True)

    print("\nread data from test.csv...")
    test_data, _ = csv_reader.read_csv(name="test.csv", train=False)

    print("\nget word ids - index dic...")
    embedding_file = "word_embedding.txt"
    new_embedding_file = "../instances/word_embed.txt"
    word_id_index, word_unk = csv_reader.get_ids_from_embeddings(embedding_file, new_embedding_file)  # 2307

    print("\nget char ids - index dic...")
    embedding_file = "char_embedding.txt"
    new_embedding_file = "../instances/char_embed.txt"
    char_id_index, char_unk = csv_reader.get_ids_from_embeddings(embedding_file, new_embedding_file)  # 2307

    print("\nread question and convert the word id and char id to index using word/char ids - index dic...")
    id_question_words, id_question_chars = csv_reader.read_questions(name="question_id.csv",
                                                                     word_id_index=word_id_index,
                                                                     char_id_index=char_id_index,
                                                                     word_unk=word_unk,
                                                                     char_unk=char_unk)

    print(train_data[0])
    print(test_data[0])

    er = ExampleReader()
    train_word_inputs1, train_word_inputs2 = er.question_pairs2question_inputs(inputs=train_data, id_questions=id_question_words)
    test_word_inputs1, test_word_inputs2 = er.question_pairs2question_inputs(inputs=test_data, id_questions=id_question_words)

    train_char_inputs1, train_char_inputs2 = er.question_pairs2question_inputs(inputs=train_data, id_questions=id_question_chars)
    test_char_inputs1, test_char_inputs2 = er.question_pairs2question_inputs(inputs=test_data, id_questions=id_question_chars)

    skf = StratifiedKFold(n_splits=5)
    inputs = np.stack([train_word_inputs1, train_word_inputs2], axis=1)
    skf.get_n_splits(inputs, train_label)
    for train_index, test_index in skf.split(inputs, train_label):
        X_train, X_test = inputs[train_index], inputs[test_index]
        y_train, y_test = train_label[train_index], train_label[test_index]
        print(str(np.shape(X_train)))
        print(str(np.shape(y_train)))
        print(str(np.shape(X_test)))
