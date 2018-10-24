
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

        self.train_question_inputs1, self.train_question_inputs2 = [None, None]
        self.load_data()

        self.Q1, self.Q2 = self.make_input()
        self.output = self.build_model()
        self.predict()


    def build_model(self):
        raise NotImplementedError

    def train_model(self):
        pass

    def predict(self):




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

    def read_model(self, file=""):
        self.build_model()
        self.model.load_weights(self.save_dir + file)

    def make_input(self):
        Q1 = Input(shape=[], dtype='int32')
        Q2 = Input(shape=[], dtype='int32')
        return Q1, Q2




