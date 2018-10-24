
from utils import PRF, print_metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.models import Model
from keras.layers import *
from preprocess.csv_reader import CsvReader
from preprocess.example_reader import ExampleReader


class BaseModel:

    def __init__(self):

        # used dirs
        self.save_dir = "../saved_models/"
        self.embedding_dir = "../instances/word_embed.txt"

        # some basic parameters of the model
        self.model = None
        self.max_len = 100
        self.num_words = 6820

        # pre-trained embeddings and their parameters.

        self.embedding_matrix = None
        self.embedding_trainable = False
        self.EMBEDDING_DIM = 300

        self.train_question_inputs1, self.train_question_inputs2, self.train_label = None, None, None
        self.test_question_inputs1, self.test_question_inputs2 = None, None
        self.load_data()

        self.Q1, self.Q2 = self.make_input()
        self.output = self.build_model()
        self.model = Model(inputs=[self.Q1, self.Q2], outputs=self.output)

    def build_model(self):
        raise NotImplementedError

    def one_train(self, epochs, batch_size,
                  train_data, train_label,
                  dev_data, dev_label):
        for e in range(epochs):
            self.model.fit(train_data, train_label, batch_size=batch_size, verbose=0,
                           validation_data=(dev_data, dev_label))
            dev_out = self.model.predict(dev_data, batch_size=2 * batch_size, verbose=0)
            metrics = PRF(dev_label.argmax(axis=1), dev_out.argmax(axis=1))
            metrics['epoch'] = e + 1
            print_metrics(metrics, metrics_type='dev')

    def train_model(self, epochs, batch_size, kfold_num=0):
        if kfold_num > 1:
            kfold = StratifiedKFold(n_splits=kfold_num, shuffle=True)
            for train_index, dev_index in kfold.split(self.train_question_inputs1, self.train_label):
                train_data = self.train_question_inputs1[train_index], self.train_question_inputs2[train_index]
                train_label = self.train_label[train_index]
                dev_data = self.train_question_inputs1[dev_index], self.train_question_inputs2[dev_index]
                dev_label = self.train_label[dev_index]

                self.one_train(epochs, batch_size,
                               train_data, train_label, dev_data, dev_label)

        else:
            train_data0, train_data1, \
            dev_data0, dev_data1, \
            train_label, dev_label = train_test_split([self.train_question_inputs1, self.train_question_inputs2,
                                                       self.train_label],
                                                      test_size=0.2,
                                                      random_state=1)
            self.one_train(epochs, batch_size,
                           [train_data0, train_data1], train_label,
                           [dev_data0, dev_data1], dev_label)



    def predict(self):
        results = self.model.predict([self.test_question_inputs1, self.test_question_inputs2], batch_size=128, verbose=0)
        res = results.argmax(axis=1)


    def save_model(self, file=""):
        self.model.save_weights(self.save_dir + file)

    def load_data(self):
        csv_reader = CsvReader()
        print("read data from train.csv...")
        train_data, train_label = csv_reader.read_csv(name="train.csv", train=True)

        print("\nread data from test.csv...")
        test_data, _ = csv_reader.read_csv(name="test.csv", train=False)

        print("\nget word ids - index dic...")
        embedding_file = "word_embed.txt"
        new_embedding_file = "../instances/word_embed.txt"
        id_index = csv_reader.get_ids_from_embeddings(embedding_file, new_embedding_file)

        print("\nread question and convert the word id to index using word ids - index dic...")
        id_question = csv_reader.read_questions(name="question.csv", word_id_index=id_index)

        er = ExampleReader()
        self.embedding_matrix = er.get_embedding_matrix("../instances/word_embed.txt")
        self.train_question_inputs1, self.train_question_inputs2 = er.question_pairs2question_inputs(inputs=train_data,
                                                                                                     id_questions=id_question)
        self.train_label = train_label

    def read_model(self, file=""):
        self.build_model()
        self.model.load_weights(self.save_dir + file)

    def make_input(self):
        Q1 = Input(shape=[self.max_len], dtype='int32')
        Q2 = Input(shape=[self.max_len], dtype='int32')
        return Q1, Q2




