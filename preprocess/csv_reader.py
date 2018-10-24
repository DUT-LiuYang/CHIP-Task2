import csv
import numpy as np


class CsvReader:

    def __init__(self):
        self.dir = "../resource/"

    def read_csv(self, name="", train=False):
        data = []
        label = []
        csv_reader = csv.reader(open(self.dir + name, encoding='utf-8'))

        for num, row in enumerate(csv_reader):

            if num == 0:
                continue

            temp = []
            for index, value in enumerate(row):
                if train:
                    if index == 0:
                        label.append(value)
                    else:
                        temp.append(value)
                else:
                    temp.append(value)

            data.append(temp[:])

        print("number of instances: " + str(len(data)) + " " + str(len(label)))

        return data, np.array(label)

    def read_questions(self, name="", word_id_index={}, index=9648):
        id_question = {}

        csv_reader = csv.reader(open(self.dir + name, encoding='utf-8'))
        count = 0
        max_len = -1

        for num, row in enumerate(csv_reader):
            if num == 0:
                continue

            id_question[row[0]] = []
            for x in row[1].split(" "):
                if x in word_id_index.keys():
                    id_question[row[0]].append(int(word_id_index[x]))
                else:
                    id_question[row[0]].append(index)

            if max_len < len(row[1].split(" ")):
                max_len = len(row[1].split(" "))

            count += 1

        print("There are " + str(count) + " questions.")  # 35268
        print(str(max_len))  # 43

        return id_question

    def get_ids_from_embeddings(self, embedding_file="", new_embedding_file=""):
        embedding_file = self.dir + embedding_file
        id_index = {}
        index = 1
        rf = open(embedding_file, 'r', encoding='utf-8')
        wf = open(new_embedding_file, 'w', encoding='utf-8')

        s = ""
        for i in range(300):
            s += str(0) + " "
        wf.write(s.strip() + "\n")

        while True:
            line = rf.readline()
            if line == "":
                break
            # print(line.split("	")[0])
            id_index[line.split("	")[0]] = index
            index += 1
            temp = line.find("	")
            wf.write(line[temp + 1:])
        rf.close()
        wf.close()
        print("There are " + str(index - 1) + " words.")  # 9647
        return id_index, index

    def convert_label_to_onehot(self, train_label=[]):
        labels = []
        for i in train_label:
            temp = [0] * 2
            if int(i) == 1:
                temp[1] = 1
            else:
                temp[0] = 1
            labels.append(temp[:])
        return np.array(labels)


if __name__ == '__main__':
    csv_reader = CsvReader()
    print("read data from train.csv...")
    train_data, train_label = csv_reader.read_csv(name="train.csv", train=True)

    print("\nread data from test.csv...")
    test_data, _ = csv_reader.read_csv(name="test.csv", train=False)

    print("\nget word ids - index dic...")
    embedding_file = "word_embedding.txt"
    new_embedding_file = "../instances/word_embed.txt"
    id_index, index = csv_reader.get_ids_from_embeddings(embedding_file, new_embedding_file)

    print("\nread question and convert the word id to index using word ids - index dic...")
    id_question = csv_reader.read_questions(name="question_id.csv", word_id_index=id_index)

