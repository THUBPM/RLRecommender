import os
import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, data_dir, log_file):
        self.log_file = log_file
        self.data_dir = data_dir

        self.entity_dict = {}
        self.relation_dict = {}
        self.graph_dict = {}
        self.n_entity = 0
        self.n_relation = 0
        self.n_graph = 0

        self.training_triples = []  # list of triples in the form of (h, t, r)
        self.validation_triples = []
        self.test_triples = []
        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0

        self.load_dict()
        self.load_triples()

        self.golden_triple_pool = set(self.training_triples) | set(self.validation_triples) | set(self.test_triples)

    def load_dict(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        graph_dict_file = 'graph2id.txt'
        print('-----Loading entity dict-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Loading entity dict-----\r\n')
        file_object.close()
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        file_object = open(self.log_file, 'a')
        file_object.write('#entity: {}'.format(self.n_entity) + '\r\n')
        file_object.write('-----Loading relation dict-----\r\n')
        file_object.close()
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.relation_dict)
        print('#relation: {}'.format(self.n_relation))
        print('-----Loading graph dict-----')
        file_object = open(self.log_file, 'a')
        file_object.write('#relation: {}'.format(self.n_relation) + '\r\n')
        file_object.write('-----Loading graph dict-----\r\n')
        file_object.close()
        graph_df = pd.read_table(os.path.join(self.data_dir, graph_dict_file), header=None)
        self.graph_dict = dict(zip(graph_df[0], graph_df[1]))
        self.n_graph = len(self.graph_dict)
        print('#graph: {}'.format(self.n_graph))
        file_object = open(self.log_file, 'a')
        file_object.write('#graph: {}'.format(self.n_graph) + '\r\n')
        file_object.close()

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'valid.txt'
        test_file = 'test.txt'
        print('-----Loading training triples-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Loading training triples-----\r\n')
        file_object.close()
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        self.training_triples = list(zip([self.entity_dict[h] for h in training_df[0]],
                                         [self.entity_dict[t] for t in training_df[1]],
                                         [self.relation_dict[r] for r in training_df[2]],
                                         [self.graph_dict[g] for g in training_df[3]]))
        self.n_training_triple = len(self.training_triples)
        print('#training triple: {}'.format(self.n_training_triple))
        print('-----Loading validation triples-----')
        file_object = open(self.log_file, 'a')
        file_object.write('#training triple: {}'.format(self.n_training_triple) + '\r\n')
        file_object.write('-----Loading validation triples-----\r\n')
        file_object.close()
        validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        self.validation_triples = list(zip([self.entity_dict[h] for h in validation_df[0]],
                                           [self.entity_dict[t] for t in validation_df[1]],
                                           [self.relation_dict[r] for r in validation_df[2]],
                                           [self.graph_dict[g] for g in validation_df[3]]))
        self.n_validation_triple = len(self.validation_triples)
        print('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples-----')
        file_object = open(self.log_file, 'a')
        file_object.write('#validation triple: {}'.format(self.n_validation_triple) + '\r\n')
        file_object.write('-----Loading test triples-----\r\n')
        file_object.close()
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        self.test_triples = list(zip([self.entity_dict[h] for h in test_df[0]],
                                     [self.entity_dict[t] for t in test_df[1]],
                                     [self.relation_dict[r] for r in test_df[2]],
                                     [self.graph_dict[g] for g in test_df[3]]))
        self.n_test_triple = len(self.test_triples)
        print('#test triple: {}'.format(self.n_test_triple))
        file_object = open(self.log_file, 'a')
        file_object.write('#test triple: {}'.format(self.n_test_triple) + '\r\n')
        file_object.close()

    def next_raw_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_training_triple)
        start = 0
        while start < self.n_training_triple:
            end = min(start + batch_size, self.n_training_triple)
            yield [self.training_triples[i] for i in rand_idx[start:end]]
            start = end

    def next_raw_eval_batch(self, eval_batch_size):
        start = 0
        while start < self.n_test_triple:
            end = min(start + eval_batch_size, self.n_test_triple)
            yield self.test_triples[start:end]
            start = end
