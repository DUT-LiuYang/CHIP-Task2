import difflib
from itertools import combinations

import numpy as np
import pandas as pd
import timeit
import networkx as nx


class FeatureReader:

    def __init__(self):
        pass

    def generate_gragh_features(self):
        df_train = pd.read_csv('../resource/train.csv').fillna("")
        df_test = pd.read_csv('../resource/test.csv').fillna("")
        len_train = df_train.shape[0]

        df = pd.concat([df_train[['qid1', 'qid2']], df_test[['qid1', 'qid2']]], axis=0)

        G = nx.Graph()
        edges = [tuple(x) for x in df[['qid1', 'qid2']].values]
        G.add_edges_from(edges)

        map_label = list(((x[0], x[1])) for x in df[['qid1', 'qid2']].values)
        map_clique_size = {}
        cliques = sorted(list(nx.find_cliques(G)), key=lambda x: len(x))
        print("start")
        # count = 0
        for cli in cliques:
            for q1, q2 in combinations(cli, 2):
                if (q1, q2) in map_label:
                    map_clique_size[q1, q2] = len(cli)
                elif (q2, q1) in map_label:
                    map_clique_size[q2, q1] = len(cli)
        print("finish")

        df['clique_size'] = df.apply(lambda row: map_clique_size.get((row['qid1'], row['qid2']), -1), axis=1)

        df[['clique_size']][:len_train].to_csv('train_feature_graph_clique.csv', index=False)
        df[['clique_size']][len_train:].to_csv('test_feature_graph_clique.csv', index=False)

    def get_magic_feature(self):
        train_orig = pd.read_csv('../resource/train.csv', header=0)
        test_orig = pd.read_csv('../resource/test.csv', header=0)

        tic0 = timeit.default_timer()
        df1 = train_orig[['qid1']].copy()
        df2 = train_orig[['qid2']].copy()
        df1_test = test_orig[['qid1']].copy()
        df2_test = test_orig[['qid2']].copy()

        df2.rename(columns={'qid2': 'qid1'}, inplace=True)
        df2_test.rename(columns={'qid2': 'qid1'}, inplace=True)

        train_questions = df1.append(df2)
        train_questions = train_questions.append(df1_test)
        train_questions = train_questions.append(df2_test)
        train_questions.drop_duplicates(subset=['qid1'], inplace=True)

        train_questions.reset_index(inplace=True, drop=True)

        questions_dict = pd.Series(train_questions.index.values, index=train_questions.qid1.values).to_dict()

        train_cp = train_orig.copy()
        test_cp = test_orig.copy()

        test_cp['label'] = -1

        comb = pd.concat([train_cp, test_cp])

        comb['q1_hash'] = comb['qid1'].map(questions_dict)
        comb['q2_hash'] = comb['qid2'].map(questions_dict)

        q1_vc = comb.q1_hash.value_counts().to_dict()
        q2_vc = comb.q2_hash.value_counts().to_dict()

        def try_apply_dict(x, dict_to_apply):
            try:
                return dict_to_apply[x]
            except KeyError:
                return 0

        comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))
        comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))

        train_comb = comb[comb['label'] >= 0][['q1_hash', 'q2_hash', 'q1_freq', 'q2_freq']]
        test_comb = comb[comb['label'] < 0][['q1_hash', 'q2_hash', 'q1_freq', 'q2_freq']]

        a = np.array(train_comb, dtype='float32')
        print(np.shape(a))
        b = np.array(test_comb, dtype='float32')
        return a, b


if __name__ == '__main__':
    f = FeatureReader()
    f.get_magic_feature()
