import math
import timeit
import random
import numpy as np
import tensorflow as tf
import multiprocessing as mp


class TransE:
    def __init__(self, dataset, embedding_dim, margin_value, score_func,
                 batch_size, eval_batch_size, learning_rate, n_generator, n_rank_calculator, log_file):
        self.log_file = log_file
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator

        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 4])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 4])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None

        '''ops for evaluation'''
        self.head_prediction = tf.placeholder(tf.int32)
        self.tail_prediction = tf.placeholder(tf.int32)
        self.relation_prediction = tf.placeholder(tf.int32)
        self.graph_prediction = tf.placeholder(tf.int32)
        self.head_prediction_raw = tf.placeholder(dtype=tf.int32, shape=[None, 4])
        self.tail_prediction_raw = tf.placeholder(dtype=tf.int32, shape=[None, 4])
        self.head_prediction_filter = tf.placeholder(dtype=tf.int32, shape=[None, 4])
        self.tail_prediction_filter = tf.placeholder(dtype=tf.int32, shape=[None, 4])
        self.score_eval = None

        bound = 6 / math.sqrt(self.embedding_dim)

        '''embeddings'''
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[dataset.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[dataset.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)

        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.score_eval = self.evaluate(self.head_prediction, self.tail_prediction, self.relation_prediction, self.graph_prediction,
                                            self.head_prediction_raw, self.tail_prediction_raw,
                                            self.head_prediction_filter, self.tail_prediction_filter)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = head_pos + relation_pos - tail_pos
            distance_neg = head_neg + relation_neg - tail_neg

        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.sqrt(tf.reduce_sum(tf.square(distance_pos), axis=1))
                score_neg = tf.sqrt(tf.reduce_sum(tf.square(distance_neg), axis=1))
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')

        return loss

    def evaluate(self, head_prediction, tail_prediction, relation_prediction, graph_prediction, head_prediction_raw, tail_prediction_raw, head_prediction_filter, tail_prediction_filter):
        with tf.name_scope('lookup'):
            '''Raw'''
            head_prediction_raw_h = tf.nn.embedding_lookup(self.entity_embedding, head_prediction_raw[:, 0])
            head_prediction_raw_t = tf.nn.embedding_lookup(self.entity_embedding, head_prediction_raw[:, 1])
            head_prediction_raw_r = tf.nn.embedding_lookup(self.relation_embedding, head_prediction_raw[:, 2])
            tail_prediction_raw_h = tf.nn.embedding_lookup(self.entity_embedding, tail_prediction_raw[:, 0])
            tail_prediction_raw_t = tf.nn.embedding_lookup(self.entity_embedding, tail_prediction_raw[:, 1])
            tail_prediction_raw_r = tf.nn.embedding_lookup(self.relation_embedding, tail_prediction_raw[:, 2])
            '''Filter'''
            head_prediction_filter_h = tf.nn.embedding_lookup(self.entity_embedding, head_prediction_filter[:, 0])
            head_prediction_filter_t = tf.nn.embedding_lookup(self.entity_embedding, head_prediction_filter[:, 1])
            head_prediction_filter_r = tf.nn.embedding_lookup(self.relation_embedding, head_prediction_filter[:, 2])
            tail_prediction_filter_h = tf.nn.embedding_lookup(self.entity_embedding, tail_prediction_filter[:, 0])
            tail_prediction_filter_t = tf.nn.embedding_lookup(self.entity_embedding, tail_prediction_filter[:, 1])
            tail_prediction_filter_r = tf.nn.embedding_lookup(self.relation_embedding, tail_prediction_filter[:, 2])
        with tf.name_scope('link'):
            distance_head_prediction_raw = head_prediction_raw_h + head_prediction_raw_r - head_prediction_raw_t
            distance_tail_prediction_raw = tail_prediction_raw_h + tail_prediction_raw_r - tail_prediction_raw_t
            distance_head_prediction_filter = head_prediction_filter_h + head_prediction_filter_r - head_prediction_filter_t
            distance_tail_prediction_filter = tail_prediction_filter_h + tail_prediction_filter_r - tail_prediction_filter_t
        with tf.name_scope('score'):
            if self.score_func == 'L1':  # L1 score
                score_head_prediction_raw = tf.reduce_sum(tf.abs(distance_head_prediction_raw), axis=1)
                score_tail_prediction_raw = tf.reduce_sum(tf.abs(distance_tail_prediction_raw), axis=1)
                score_head_prediction_filter = tf.reduce_sum(tf.abs(distance_head_prediction_filter), axis=1)
                score_tail_prediction_filter = tf.reduce_sum(tf.abs(distance_tail_prediction_filter), axis=1)
            else:  # L2 score
                score_head_prediction_raw = tf.sqrt(tf.reduce_sum(tf.square(distance_head_prediction_raw), axis=1))
                score_tail_prediction_raw = tf.sqrt(tf.reduce_sum(tf.square(distance_tail_prediction_raw), axis=1))
                score_head_prediction_filter = tf.sqrt(tf.reduce_sum(tf.square(distance_head_prediction_filter), axis=1))
                score_tail_prediction_filter = tf.sqrt(tf.reduce_sum(tf.square(distance_tail_prediction_filter), axis=1))
        return head_prediction, tail_prediction, relation_prediction, graph_prediction, \
               score_head_prediction_raw, score_tail_prediction_raw, score_head_prediction_filter, score_tail_prediction_filter

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                    'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Start training-----\r\n')
        file_object.close()
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.dataset.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Constructing training batches-----\r\n')
        file_object.close()
        epoch_loss = 0
        n_used_triple = 0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            batch_loss, _, summary = session.run(fetches=[self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.8f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.dataset.n_training_triple,
                                                                            batch_loss / len(batch_pos)))
            file_object = open(self.log_file, 'a')
            file_object.write('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.8f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.dataset.n_training_triple,
                                                                            batch_loss / len(batch_pos)) + '\r\n')
            file_object.close()
        print()
        print('mean loss: {:.8f}'.format(epoch_loss / self.dataset.n_training_triple))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish training-----')
        print('-----Check norm-----')
        file_object = open(self.log_file, 'a')
        file_object.write('\r\n')
        file_object.write('mean loss: {:.8f}'.format(epoch_loss / self.dataset.n_training_triple) + '\r\n')
        file_object.write('cost time: {:.3f}s'.format(timeit.default_timer() - start) + '\r\n')
        file_object.write('-----Finish training-----\r\n')
        file_object.write('-----Check norm-----\r\n')
        file_object.close()
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))
        file_object = open(self.log_file, 'a')
        file_object.write('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm) + '\r\n')
        file_object.close()

    def generate_training_batch(self, in_queue, out_queue):
        while True:
            raw_batch = in_queue.get()
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                corrupt_head_prob = np.random.binomial(1, 0.5)
                for head, tail, relation, graph in batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            head_neg = random.sample(list(self.dataset.entity_dict.values()), 1)[0]
                        else:
                            tail_neg = random.sample(list(self.dataset.entity_dict.values()), 1)[0]
                        if (head_neg, tail_neg, relation, graph) not in self.dataset.golden_triple_pool:
                            break
                    batch_neg.append((head_neg, tail_neg, relation, graph))
                out_queue.put((batch_pos, batch_neg))

    def launch_evaluation(self, session):
        raw_eval_batch_queue = mp.Queue()
        eval_batch_queue = mp.Queue(10000)
        eval_batch_and_score_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.generate_evaluation_batch, kwargs={'in_queue': raw_eval_batch_queue,
                                                                      'out_queue': eval_batch_queue}).start()
        print('-----Start evaluation-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Start evaluation-----\r\n')
        file_object.close()
        start = timeit.default_timer()
        n_eval_triple = 0
        for raw_eval_batch in self.dataset.next_raw_eval_batch(self.eval_batch_size):
            raw_eval_batch_queue.put(raw_eval_batch)
            n_eval_triple += len(raw_eval_batch)
        print('#eval triple: {}'.format(n_eval_triple))
        file_object = open(self.log_file, 'a')
        file_object.write('#eval triple: {}'.format(n_eval_triple) + '\r\n')
        file_object.close()
        for _ in range(self.n_generator):
            raw_eval_batch_queue.put(None)
        print('-----Constructing evaluation batches-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Constructing evaluation batches-----\r\n')
        file_object.close()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_batch_and_score_queue,
                                                           'out_queue': rank_result_queue}).start()
        for i in range(n_eval_triple):
            head_prediction, tail_prediction, relation_prediction, graph_prediction, \
                head_prediction_batch_raw, tail_prediction_batch_raw, \
                head_prediction_batch_filter, tail_prediction_batch_filter = eval_batch_queue.get()
            score_eval = session.run(fetches=self.score_eval,
                                     feed_dict={self.head_prediction: head_prediction,
                                                self.tail_prediction: tail_prediction,
                                                self.relation_prediction: relation_prediction,
                                                self.graph_prediction: graph_prediction,
                                                self.head_prediction_raw: head_prediction_batch_raw,
                                                self.tail_prediction_raw: tail_prediction_batch_raw,
                                                self.head_prediction_filter: head_prediction_batch_filter,
                                                self.tail_prediction_filter: tail_prediction_batch_filter})
            head_prediction, tail_prediction, relation_prediction, graph_prediction, \
                head_prediction_score_raw, tail_prediction_score_raw, \
                head_prediction_score_filter, tail_prediction_score_filter = score_eval
            eval_batch_and_score_queue.put((head_prediction, tail_prediction, relation_prediction, graph_prediction,
                                            head_prediction_score_raw, tail_prediction_score_raw,
                                            head_prediction_score_filter, tail_prediction_score_filter))
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               i + 1,
                                                               n_eval_triple))
            file_object = open(self.log_file, 'a')
            file_object.write('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               i + 1,
                                                               n_eval_triple) + '\r\n')
            file_object.close()
        print()
        file_object = open(self.log_file, 'a')
        file_object.write('\r\n')
        file_object.close()
        for _ in range(self.n_rank_calculator):
            eval_batch_and_score_queue.put(None)
        print('-----Joining all rank calculator-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Joining all rank calculator-----\r\n')
        file_object.close()
        eval_batch_and_score_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----All rank calculation accomplished-----\r\n')
        file_object.write('-----Obtaining evaluation results-----\r\n')
        file_object.close()
        head_rank_raw_sum = 0
        head_hits1_raw_sum, head_hits2_raw_sum, head_hits3_raw_sum, head_hits4_raw_sum, head_hits5_raw_sum, head_hits10_raw_sum = 0, 0, 0, 0, 0, 0
        head_r1_raw_sum, head_r2_raw_sum, head_r3_raw_sum, head_r4_raw_sum, head_r5_raw_sum, head_r10_raw_sum = 0, 0, 0, 0, 0, 0
        head_t1_raw_sum, head_t2_raw_sum, head_t3_raw_sum, head_t4_raw_sum, head_t5_raw_sum, head_t10_raw_sum = 0, 0, 0, 0, 0, 0
        head_rt1_raw_sum, head_rt2_raw_sum, head_rt3_raw_sum, head_rt4_raw_sum, head_rt5_raw_sum, head_rt10_raw_sum = 0, 0, 0, 0, 0, 0
        tail_rank_raw_sum = 0
        tail_hits1_raw_sum, tail_hits2_raw_sum, tail_hits3_raw_sum, tail_hits4_raw_sum, tail_hits5_raw_sum, tail_hits10_raw_sum = 0, 0, 0, 0, 0, 0
        tail_r1_raw_sum, tail_r2_raw_sum, tail_r3_raw_sum, tail_r4_raw_sum, tail_r5_raw_sum, tail_r10_raw_sum = 0, 0, 0, 0, 0, 0
        tail_t1_raw_sum, tail_t2_raw_sum, tail_t3_raw_sum, tail_t4_raw_sum, tail_t5_raw_sum, tail_t10_raw_sum = 0, 0, 0, 0, 0, 0
        tail_rt1_raw_sum, tail_rt2_raw_sum, tail_rt3_raw_sum, tail_rt4_raw_sum, tail_rt5_raw_sum, tail_rt10_raw_sum = 0, 0, 0, 0, 0, 0
        head_rank_filter_sum = 0
        head_hits1_filter_sum, head_hits2_filter_sum, head_hits3_filter_sum, head_hits4_filter_sum, head_hits5_filter_sum, head_hits10_filter_sum = 0, 0, 0, 0, 0, 0
        head_r1_filter_sum, head_r2_filter_sum, head_r3_filter_sum, head_r4_filter_sum, head_r5_filter_sum, head_r10_filter_sum = 0, 0, 0, 0, 0, 0
        head_t1_filter_sum, head_t2_filter_sum, head_t3_filter_sum, head_t4_filter_sum, head_t5_filter_sum, head_t10_filter_sum = 0, 0, 0, 0, 0, 0
        head_rt1_filter_sum, head_rt2_filter_sum, head_rt3_filter_sum, head_rt4_filter_sum, head_rt5_filter_sum, head_rt10_filter_sum = 0, 0, 0, 0, 0, 0
        tail_rank_filter_sum = 0
        tail_hits1_filter_sum, tail_hits2_filter_sum, tail_hits3_filter_sum, tail_hits4_filter_sum, tail_hits5_filter_sum, tail_hits10_filter_sum = 0, 0, 0, 0, 0, 0
        tail_r1_filter_sum, tail_r2_filter_sum, tail_r3_filter_sum, tail_r4_filter_sum, tail_r5_filter_sum, tail_r10_filter_sum = 0, 0, 0, 0, 0, 0
        tail_t1_filter_sum, tail_t2_filter_sum, tail_t3_filter_sum, tail_t4_filter_sum, tail_t5_filter_sum, tail_t10_filter_sum = 0, 0, 0, 0, 0, 0
        tail_rt1_filter_sum, tail_rt2_filter_sum, tail_rt3_filter_sum, tail_rt4_filter_sum, tail_rt5_filter_sum, tail_rt10_filter_sum = 0, 0, 0, 0, 0, 0
        for _ in range(n_eval_triple):
            head_rank_raw, head_hits1_raw, head_hits2_raw, head_hits3_raw, head_hits4_raw, head_hits5_raw, head_hits10_raw, \
                head_r1_raw, head_r2_raw, head_r3_raw, head_r4_raw, head_r5_raw, head_r10_raw, \
                head_t1_raw, head_t2_raw, head_t3_raw, head_t4_raw, head_t5_raw, head_t10_raw, \
                head_rt1_raw, head_rt2_raw, head_rt3_raw, head_rt4_raw, head_rt5_raw, head_rt10_raw, \
                tail_rank_raw, tail_hits1_raw, tail_hits2_raw, tail_hits3_raw, tail_hits4_raw, tail_hits5_raw, tail_hits10_raw, \
                tail_r1_raw, tail_r2_raw, tail_r3_raw, tail_r4_raw, tail_r5_raw, tail_r10_raw, \
                tail_t1_raw, tail_t2_raw, tail_t3_raw, tail_t4_raw, tail_t5_raw, tail_t10_raw, \
                tail_rt1_raw, tail_rt2_raw, tail_rt3_raw, tail_rt4_raw, tail_rt5_raw, tail_rt10_raw, \
                head_rank_filter, head_hits1_filter, head_hits2_filter, head_hits3_filter, head_hits4_filter, head_hits5_filter, head_hits10_filter, \
                head_r1_filter, head_r2_filter, head_r3_filter, head_r4_filter, head_r5_filter, head_r10_filter, \
                head_t1_filter, head_t2_filter, head_t3_filter, head_t4_filter, head_t5_filter, head_t10_filter, \
                head_rt1_filter, head_rt2_filter, head_rt3_filter, head_rt4_filter, head_rt5_filter, head_rt10_filter, \
                tail_rank_filter, tail_hits1_filter, tail_hits2_filter, tail_hits3_filter, tail_hits4_filter, tail_hits5_filter, tail_hits10_filter, \
                tail_r1_filter, tail_r2_filter, tail_r3_filter, tail_r4_filter, tail_r5_filter, tail_r10_filter, \
                tail_t1_filter, tail_t2_filter, tail_t3_filter, tail_t4_filter, tail_t5_filter, tail_t10_filter, \
                tail_rt1_filter, tail_rt2_filter, tail_rt3_filter, tail_rt4_filter, tail_rt5_filter, tail_rt10_filter = rank_result_queue.get()
            head_rank_raw_sum += head_rank_raw
            head_hits1_raw_sum += head_hits1_raw
            head_hits2_raw_sum += head_hits2_raw
            head_hits3_raw_sum += head_hits3_raw
            head_hits4_raw_sum += head_hits4_raw
            head_hits5_raw_sum += head_hits5_raw
            head_hits10_raw_sum += head_hits10_raw
            head_r1_raw_sum += head_r1_raw
            head_r2_raw_sum += head_r2_raw
            head_r3_raw_sum += head_r3_raw
            head_r4_raw_sum += head_r4_raw
            head_r5_raw_sum += head_r5_raw
            head_r10_raw_sum += head_r10_raw
            head_t1_raw_sum += head_t1_raw
            head_t2_raw_sum += head_t2_raw
            head_t3_raw_sum += head_t3_raw
            head_t4_raw_sum += head_t4_raw
            head_t5_raw_sum += head_t5_raw
            head_t10_raw_sum += head_t10_raw
            head_rt1_raw_sum += head_rt1_raw
            head_rt2_raw_sum += head_rt2_raw
            head_rt3_raw_sum += head_rt3_raw
            head_rt4_raw_sum += head_rt4_raw
            head_rt5_raw_sum += head_rt5_raw
            head_rt10_raw_sum += head_rt10_raw
            tail_rank_raw_sum += tail_rank_raw
            tail_hits1_raw_sum += tail_hits1_raw
            tail_hits2_raw_sum += tail_hits2_raw
            tail_hits3_raw_sum += tail_hits3_raw
            tail_hits4_raw_sum += tail_hits4_raw
            tail_hits5_raw_sum += tail_hits5_raw
            tail_hits10_raw_sum += tail_hits10_raw
            tail_r1_raw_sum += tail_r1_raw
            tail_r2_raw_sum += tail_r2_raw
            tail_r3_raw_sum += tail_r3_raw
            tail_r4_raw_sum += tail_r4_raw
            tail_r5_raw_sum += tail_r5_raw
            tail_r10_raw_sum += tail_r10_raw
            tail_t1_raw_sum += tail_t1_raw
            tail_t2_raw_sum += tail_t2_raw
            tail_t3_raw_sum += tail_t3_raw
            tail_t4_raw_sum += tail_t4_raw
            tail_t5_raw_sum += tail_t5_raw
            tail_t10_raw_sum += tail_t10_raw
            tail_rt1_raw_sum += tail_rt1_raw
            tail_rt2_raw_sum += tail_rt2_raw
            tail_rt3_raw_sum += tail_rt3_raw
            tail_rt4_raw_sum += tail_rt4_raw
            tail_rt5_raw_sum += tail_rt5_raw
            tail_rt10_raw_sum += tail_rt10_raw
            head_rank_filter_sum += head_rank_filter
            head_hits1_filter_sum += head_hits1_filter
            head_hits2_filter_sum += head_hits2_filter
            head_hits3_filter_sum += head_hits3_filter
            head_hits4_filter_sum += head_hits4_filter
            head_hits5_filter_sum += head_hits5_filter
            head_hits10_filter_sum += head_hits10_filter
            head_r1_filter_sum += head_r1_filter
            head_r2_filter_sum += head_r2_filter
            head_r3_filter_sum += head_r3_filter
            head_r4_filter_sum += head_r4_filter
            head_r5_filter_sum += head_r5_filter
            head_r10_filter_sum += head_r10_filter
            head_t1_filter_sum += head_t1_filter
            head_t2_filter_sum += head_t2_filter
            head_t3_filter_sum += head_t3_filter
            head_t4_filter_sum += head_t4_filter
            head_t5_filter_sum += head_t5_filter
            head_t10_filter_sum += head_t10_filter
            head_rt1_filter_sum += head_rt1_filter
            head_rt2_filter_sum += head_rt2_filter
            head_rt3_filter_sum += head_rt3_filter
            head_rt4_filter_sum += head_rt4_filter
            head_rt5_filter_sum += head_rt5_filter
            head_rt10_filter_sum += head_rt10_filter
            tail_rank_filter_sum += tail_rank_filter
            tail_hits1_filter_sum += tail_hits1_filter
            tail_hits2_filter_sum += tail_hits2_filter
            tail_hits3_filter_sum += tail_hits3_filter
            tail_hits4_filter_sum += tail_hits4_filter
            tail_hits5_filter_sum += tail_hits5_filter
            tail_hits10_filter_sum += tail_hits10_filter
            tail_r1_filter_sum += tail_r1_filter
            tail_r2_filter_sum += tail_r2_filter
            tail_r3_filter_sum += tail_r3_filter
            tail_r4_filter_sum += tail_r4_filter
            tail_r5_filter_sum += tail_r5_filter
            tail_r10_filter_sum += tail_r10_filter
            tail_t1_filter_sum += tail_t1_filter
            tail_t2_filter_sum += tail_t2_filter
            tail_t3_filter_sum += tail_t3_filter
            tail_t4_filter_sum += tail_t4_filter
            tail_t5_filter_sum += tail_t5_filter
            tail_t10_filter_sum += tail_t10_filter
            tail_rt1_filter_sum += tail_rt1_filter
            tail_rt2_filter_sum += tail_rt2_filter
            tail_rt3_filter_sum += tail_rt3_filter
            tail_rt4_filter_sum += tail_rt4_filter
            tail_rt5_filter_sum += tail_rt5_filter
            tail_rt10_filter_sum += tail_rt10_filter
        print('-----Raw-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Raw-----\r\n')
        file_object.close()
        '''raw'''
        head_meanrank_raw = 1.0 * head_rank_raw_sum / n_eval_triple
        head_hits1_raw = 1.0 * head_hits1_raw_sum / n_eval_triple
        head_hits2_raw = 1.0 * head_hits2_raw_sum / n_eval_triple
        head_hits3_raw = 1.0 * head_hits3_raw_sum / n_eval_triple
        head_hits4_raw = 1.0 * head_hits4_raw_sum / n_eval_triple
        head_hits5_raw = 1.0 * head_hits5_raw_sum / n_eval_triple
        head_hits10_raw = 1.0 * head_hits10_raw_sum / n_eval_triple
        head_precision1_raw = 1.0 * head_rt1_raw_sum / head_r1_raw_sum
        head_precision2_raw = 1.0 * head_rt2_raw_sum / head_r2_raw_sum
        head_precision3_raw = 1.0 * head_rt3_raw_sum / head_r3_raw_sum
        head_precision4_raw = 1.0 * head_rt4_raw_sum / head_r4_raw_sum
        head_precision5_raw = 1.0 * head_rt5_raw_sum / head_r5_raw_sum
        head_precision10_raw = 1.0 * head_rt10_raw_sum / head_r10_raw_sum
        head_recall1_raw = 1.0 * head_rt1_raw_sum / head_t1_raw_sum
        head_recall2_raw = 1.0 * head_rt2_raw_sum / head_t2_raw_sum
        head_recall3_raw = 1.0 * head_rt3_raw_sum / head_t3_raw_sum
        head_recall4_raw = 1.0 * head_rt4_raw_sum / head_t4_raw_sum
        head_recall5_raw = 1.0 * head_rt5_raw_sum / head_t5_raw_sum
        head_recall10_raw = 1.0 * head_rt10_raw_sum / head_t10_raw_sum
        head_f11_raw = 2.0 * head_precision1_raw * head_recall1_raw / (head_precision1_raw + head_recall1_raw) if head_precision1_raw + head_recall1_raw > 0 else 0
        head_f12_raw = 2.0 * head_precision2_raw * head_recall2_raw / (head_precision2_raw + head_recall2_raw) if head_precision2_raw + head_recall2_raw > 0 else 0
        head_f13_raw = 2.0 * head_precision3_raw * head_recall3_raw / (head_precision3_raw + head_recall3_raw) if head_precision3_raw + head_recall3_raw > 0 else 0
        head_f14_raw = 2.0 * head_precision4_raw * head_recall4_raw / (head_precision4_raw + head_recall4_raw) if head_precision4_raw + head_recall4_raw > 0 else 0
        head_f15_raw = 2.0 * head_precision5_raw * head_recall5_raw / (head_precision5_raw + head_recall5_raw) if head_precision5_raw + head_recall5_raw > 0 else 0
        head_f110_raw = 2.0 * head_precision10_raw * head_recall10_raw / (head_precision10_raw + head_recall10_raw) if head_precision10_raw + head_recall10_raw > 0 else 0
        tail_meanrank_raw = 1.0 * tail_rank_raw_sum / n_eval_triple
        tail_hits1_raw = 1.0 * tail_hits1_raw_sum / n_eval_triple
        tail_hits2_raw = 1.0 * tail_hits2_raw_sum / n_eval_triple
        tail_hits3_raw = 1.0 * tail_hits3_raw_sum / n_eval_triple
        tail_hits4_raw = 1.0 * tail_hits4_raw_sum / n_eval_triple
        tail_hits5_raw = 1.0 * tail_hits5_raw_sum / n_eval_triple
        tail_hits10_raw = 1.0 * tail_hits10_raw_sum / n_eval_triple
        tail_precision1_raw = 1.0 * tail_rt1_raw_sum / tail_r1_raw_sum
        tail_precision2_raw = 1.0 * tail_rt2_raw_sum / tail_r2_raw_sum
        tail_precision3_raw = 1.0 * tail_rt3_raw_sum / tail_r3_raw_sum
        tail_precision4_raw = 1.0 * tail_rt4_raw_sum / tail_r4_raw_sum
        tail_precision5_raw = 1.0 * tail_rt5_raw_sum / tail_r5_raw_sum
        tail_precision10_raw = 1.0 * tail_rt10_raw_sum / tail_r10_raw_sum
        tail_recall1_raw = 1.0 * tail_rt1_raw_sum / tail_t1_raw_sum
        tail_recall2_raw = 1.0 * tail_rt2_raw_sum / tail_t2_raw_sum
        tail_recall3_raw = 1.0 * tail_rt3_raw_sum / tail_t3_raw_sum
        tail_recall4_raw = 1.0 * tail_rt4_raw_sum / tail_t4_raw_sum
        tail_recall5_raw = 1.0 * tail_rt5_raw_sum / tail_t5_raw_sum
        tail_recall10_raw = 1.0 * tail_rt10_raw_sum / tail_t10_raw_sum
        tail_f11_raw = 2.0 * tail_precision1_raw * tail_recall1_raw / (tail_precision1_raw + tail_recall1_raw) if tail_precision1_raw + tail_recall1_raw > 0 else 0
        tail_f12_raw = 2.0 * tail_precision2_raw * tail_recall2_raw / (tail_precision2_raw + tail_recall2_raw) if tail_precision2_raw + tail_recall2_raw > 0 else 0
        tail_f13_raw = 2.0 * tail_precision3_raw * tail_recall3_raw / (tail_precision3_raw + tail_recall3_raw) if tail_precision3_raw + tail_recall3_raw > 0 else 0
        tail_f14_raw = 2.0 * tail_precision4_raw * tail_recall4_raw / (tail_precision4_raw + tail_recall4_raw) if tail_precision4_raw + tail_recall4_raw > 0 else 0
        tail_f15_raw = 2.0 * tail_precision5_raw * tail_recall5_raw / (tail_precision5_raw + tail_recall5_raw) if tail_precision5_raw + tail_recall5_raw > 0 else 0
        tail_f110_raw = 2.0 * tail_precision10_raw * tail_recall10_raw / (tail_precision10_raw + tail_recall10_raw) if tail_precision10_raw + tail_recall10_raw > 0 else 0
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}'.format(head_meanrank_raw))
        print('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format(head_hits1_raw, head_hits2_raw, head_hits3_raw, head_hits4_raw, head_hits5_raw, head_hits10_raw))
        print('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format(head_precision1_raw, head_precision2_raw, head_precision3_raw, head_precision4_raw, head_precision5_raw, head_precision10_raw))
        print('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format(head_recall1_raw, head_recall2_raw, head_recall3_raw, head_recall4_raw, head_recall5_raw, head_recall10_raw))
        print('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format(head_f11_raw, head_f12_raw, head_f13_raw, head_f14_raw, head_f15_raw, head_f110_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}'.format(tail_meanrank_raw))
        print('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format(tail_hits1_raw, tail_hits2_raw, tail_hits3_raw, tail_hits4_raw, tail_hits5_raw, tail_hits10_raw))
        print('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format(tail_precision1_raw, tail_precision2_raw, tail_precision3_raw, tail_precision4_raw, tail_precision5_raw, tail_precision10_raw))
        print('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format(tail_recall1_raw, tail_recall2_raw, tail_recall3_raw, tail_recall4_raw, tail_recall5_raw, tail_recall10_raw))
        print('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format(tail_f11_raw, tail_f12_raw, tail_f13_raw, tail_f14_raw, tail_f15_raw, tail_f110_raw))
        print('------Average------')
        print('MeanRank: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2))
        print('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format((head_hits1_raw + tail_hits1_raw) / 2, (head_hits2_raw + tail_hits2_raw) / 2, (head_hits3_raw + tail_hits3_raw) / 2, (head_hits4_raw + tail_hits4_raw) / 2, (head_hits5_raw + tail_hits5_raw) / 2, (head_hits10_raw + tail_hits10_raw) / 2))
        print('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format((head_precision1_raw + tail_precision1_raw) / 2, (head_precision2_raw + tail_precision2_raw) / 2, (head_precision3_raw + tail_precision3_raw) / 2, (head_precision4_raw + tail_precision4_raw) / 2, (head_precision5_raw + tail_precision5_raw) / 2, (head_precision10_raw + tail_precision10_raw) / 2))
        print('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format((head_recall1_raw + tail_recall1_raw) / 2, (head_recall2_raw + tail_recall2_raw) / 2, (head_recall3_raw + tail_recall3_raw) / 2, (head_recall4_raw + tail_recall4_raw) / 2, (head_recall5_raw + tail_recall5_raw) / 2, (head_recall10_raw + tail_recall10_raw) / 2))
        print('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format((head_f11_raw + tail_f11_raw) / 2, (head_f12_raw + tail_f12_raw) / 2, (head_f13_raw + tail_f13_raw) / 2, (head_f14_raw + tail_f14_raw) / 2, (head_f15_raw + tail_f15_raw) / 2, (head_f110_raw + tail_f110_raw) / 2))
        print('-----Filter-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Head prediction-----\r\n')
        file_object.write('MeanRank: {:.3f}'.format(head_meanrank_raw) + '\r\n')
        file_object.write('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format(head_hits1_raw, head_hits2_raw, head_hits3_raw, head_hits4_raw, head_hits5_raw, head_hits10_raw) + '\r\n')
        file_object.write('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format(head_precision1_raw, head_precision2_raw, head_precision3_raw, head_precision4_raw, head_precision5_raw, head_precision10_raw) + '\r\n')
        file_object.write('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format(head_recall1_raw, head_recall2_raw, head_recall3_raw, head_recall4_raw, head_recall5_raw, head_recall10_raw) + '\r\n')
        file_object.write('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format(head_f11_raw, head_f12_raw, head_f13_raw, head_f14_raw, head_f15_raw, head_f110_raw) + '\r\n')
        file_object.write('-----Tail prediction-----\r\n')
        file_object.write('MeanRank: {:.3f}'.format(tail_meanrank_raw) + '\r\n')
        file_object.write('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format(tail_hits1_raw, tail_hits2_raw, tail_hits3_raw, tail_hits4_raw, tail_hits5_raw, tail_hits10_raw) + '\r\n')
        file_object.write('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format(tail_precision1_raw, tail_precision2_raw, tail_precision3_raw, tail_precision4_raw, tail_precision5_raw, tail_precision10_raw) + '\r\n')
        file_object.write('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format(tail_recall1_raw, tail_recall2_raw, tail_recall3_raw, tail_recall4_raw, tail_recall5_raw, tail_recall10_raw) + '\r\n')
        file_object.write('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format(tail_f11_raw, tail_f12_raw, tail_f13_raw, tail_f14_raw, tail_f15_raw, tail_f110_raw) + '\r\n')
        file_object.write('------Average------\r\n')
        file_object.write('MeanRank: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2) + '\r\n')
        file_object.write('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format((head_hits1_raw + tail_hits1_raw) / 2, (head_hits2_raw + tail_hits2_raw) / 2, (head_hits3_raw + tail_hits3_raw) / 2, (head_hits4_raw + tail_hits4_raw) / 2, (head_hits5_raw + tail_hits5_raw) / 2, (head_hits10_raw + tail_hits10_raw) / 2) + '\r\n')
        file_object.write('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format((head_precision1_raw + tail_precision1_raw) / 2, (head_precision2_raw + tail_precision2_raw) / 2, (head_precision3_raw + tail_precision3_raw) / 2, (head_precision4_raw + tail_precision4_raw) / 2, (head_precision5_raw + tail_precision5_raw) / 2, (head_precision10_raw + tail_precision10_raw) / 2) + '\r\n')
        file_object.write('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format((head_recall1_raw + tail_recall1_raw) / 2, (head_recall2_raw + tail_recall2_raw) / 2, (head_recall3_raw + tail_recall3_raw) / 2, (head_recall4_raw + tail_recall4_raw) / 2, (head_recall5_raw + tail_recall5_raw) / 2, (head_recall10_raw + tail_recall10_raw) / 2) + '\r\n')
        file_object.write('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format((head_f11_raw + tail_f11_raw) / 2, (head_f12_raw + tail_f12_raw) / 2, (head_f13_raw + tail_f13_raw) / 2, (head_f14_raw + tail_f14_raw) / 2, (head_f15_raw + tail_f15_raw) / 2, (head_f110_raw + tail_f110_raw) / 2) + '\r\n')
        file_object.write('-----Filter-----\r\n')
        file_object.close()

        '''filter'''
        head_meanrank_filter = 1.0 * head_rank_filter_sum / n_eval_triple
        head_hits1_filter = 1.0 * head_hits1_filter_sum / n_eval_triple
        head_hits2_filter = 1.0 * head_hits2_filter_sum / n_eval_triple
        head_hits3_filter = 1.0 * head_hits3_filter_sum / n_eval_triple
        head_hits4_filter = 1.0 * head_hits4_filter_sum / n_eval_triple
        head_hits5_filter = 1.0 * head_hits5_filter_sum / n_eval_triple
        head_hits10_filter = 1.0 * head_hits10_filter_sum / n_eval_triple
        head_precision1_filter = 1.0 * head_rt1_filter_sum / head_r1_filter_sum
        head_precision2_filter = 1.0 * head_rt2_filter_sum / head_r2_filter_sum
        head_precision3_filter = 1.0 * head_rt3_filter_sum / head_r3_filter_sum
        head_precision4_filter = 1.0 * head_rt4_filter_sum / head_r4_filter_sum
        head_precision5_filter = 1.0 * head_rt5_filter_sum / head_r5_filter_sum
        head_precision10_filter = 1.0 * head_rt10_filter_sum / head_r10_filter_sum
        head_recall1_filter = 1.0 * head_rt1_filter_sum / head_t1_filter_sum
        head_recall2_filter = 1.0 * head_rt2_filter_sum / head_t2_filter_sum
        head_recall3_filter = 1.0 * head_rt3_filter_sum / head_t3_filter_sum
        head_recall4_filter = 1.0 * head_rt4_filter_sum / head_t4_filter_sum
        head_recall5_filter = 1.0 * head_rt5_filter_sum / head_t5_filter_sum
        head_recall10_filter = 1.0 * head_rt10_filter_sum / head_t10_filter_sum
        head_f11_filter = 2.0 * head_precision1_filter * head_recall1_filter / (head_precision1_filter + head_recall1_filter) if head_precision1_filter + head_recall1_filter > 0 else 0
        head_f12_filter = 2.0 * head_precision2_filter * head_recall2_filter / (head_precision2_filter + head_recall2_filter) if head_precision2_filter + head_recall2_filter > 0 else 0
        head_f13_filter = 2.0 * head_precision3_filter * head_recall3_filter / (head_precision3_filter + head_recall3_filter) if head_precision3_filter + head_recall3_filter > 0 else 0
        head_f14_filter = 2.0 * head_precision4_filter * head_recall4_filter / (head_precision4_filter + head_recall4_filter) if head_precision4_filter + head_recall4_filter > 0 else 0
        head_f15_filter = 2.0 * head_precision5_filter * head_recall5_filter / (head_precision5_filter + head_recall5_filter) if head_precision5_filter + head_recall5_filter > 0 else 0
        head_f110_filter = 2.0 * head_precision10_filter * head_recall10_filter / (head_precision10_filter + head_recall10_filter) if head_precision10_filter + head_recall10_filter > 0 else 0
        tail_meanrank_filter = 1.0 * tail_rank_filter_sum / n_eval_triple
        tail_hits1_filter = 1.0 * tail_hits1_filter_sum / n_eval_triple
        tail_hits2_filter = 1.0 * tail_hits2_filter_sum / n_eval_triple
        tail_hits3_filter = 1.0 * tail_hits3_filter_sum / n_eval_triple
        tail_hits4_filter = 1.0 * tail_hits4_filter_sum / n_eval_triple
        tail_hits5_filter = 1.0 * tail_hits5_filter_sum / n_eval_triple
        tail_hits10_filter = 1.0 * tail_hits10_filter_sum / n_eval_triple
        tail_precision1_filter = 1.0 * tail_rt1_filter_sum / tail_r1_filter_sum
        tail_precision2_filter = 1.0 * tail_rt2_filter_sum / tail_r2_filter_sum
        tail_precision3_filter = 1.0 * tail_rt3_filter_sum / tail_r3_filter_sum
        tail_precision4_filter = 1.0 * tail_rt4_filter_sum / tail_r4_filter_sum
        tail_precision5_filter = 1.0 * tail_rt5_filter_sum / tail_r5_filter_sum
        tail_precision10_filter = 1.0 * tail_rt10_filter_sum / tail_r10_filter_sum
        tail_recall1_filter = 1.0 * tail_rt1_filter_sum / tail_t1_filter_sum
        tail_recall2_filter = 1.0 * tail_rt2_filter_sum / tail_t2_filter_sum
        tail_recall3_filter = 1.0 * tail_rt3_filter_sum / tail_t3_filter_sum
        tail_recall4_filter = 1.0 * tail_rt4_filter_sum / tail_t4_filter_sum
        tail_recall5_filter = 1.0 * tail_rt5_filter_sum / tail_t5_filter_sum
        tail_recall10_filter = 1.0 * tail_rt10_filter_sum / tail_t10_filter_sum
        tail_f11_filter = 2.0 * tail_precision1_filter * tail_recall1_filter / (tail_precision1_filter + tail_recall1_filter) if tail_precision1_filter + tail_recall1_filter > 0 else 0
        tail_f12_filter = 2.0 * tail_precision2_filter * tail_recall2_filter / (tail_precision2_filter + tail_recall2_filter) if tail_precision2_filter + tail_recall2_filter > 0 else 0
        tail_f13_filter = 2.0 * tail_precision3_filter * tail_recall3_filter / (tail_precision3_filter + tail_recall3_filter) if tail_precision3_filter + tail_recall3_filter > 0 else 0
        tail_f14_filter = 2.0 * tail_precision4_filter * tail_recall4_filter / (tail_precision4_filter + tail_recall4_filter) if tail_precision4_filter + tail_recall4_filter > 0 else 0
        tail_f15_filter = 2.0 * tail_precision5_filter * tail_recall5_filter / (tail_precision5_filter + tail_recall5_filter) if tail_precision5_filter + tail_recall5_filter > 0 else 0
        tail_f110_filter = 2.0 * tail_precision10_filter * tail_recall10_filter / (tail_precision10_filter + tail_recall10_filter) if tail_precision10_filter + tail_recall10_filter > 0 else 0
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}'.format(head_meanrank_filter))
        print('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format(head_hits1_filter, head_hits2_filter, head_hits3_filter, head_hits4_filter, head_hits5_filter, head_hits10_filter))
        print('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format(head_precision1_filter, head_precision2_filter, head_precision3_filter, head_precision4_filter, head_precision5_filter, head_precision10_filter))
        print('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format(head_recall1_filter, head_recall2_filter, head_recall3_filter, head_recall4_filter, head_recall5_filter, head_recall10_filter))
        print('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format(head_f11_filter, head_f12_filter, head_f13_filter, head_f14_filter, head_f15_filter, head_f110_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}'.format(tail_meanrank_filter))
        print('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format(tail_hits1_filter, tail_hits2_filter, tail_hits3_filter, tail_hits4_filter, tail_hits5_filter, tail_hits10_filter))
        print('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format(tail_precision1_filter, tail_precision2_filter, tail_precision3_filter, tail_precision4_filter, tail_precision5_filter, tail_precision10_filter))
        print('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format(tail_recall1_filter, tail_recall2_filter, tail_recall3_filter, tail_recall4_filter, tail_recall5_filter, tail_recall10_filter))
        print('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format(tail_f11_filter, tail_f12_filter, tail_f13_filter, tail_f14_filter, tail_f15_filter, tail_f110_filter))
        print('------Average------')
        print('MeanRank: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2))
        print('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format((head_hits1_filter + tail_hits1_filter) / 2, (head_hits2_filter + tail_hits2_filter) / 2, (head_hits3_filter + tail_hits3_filter) / 2, (head_hits4_filter + tail_hits4_filter) / 2, (head_hits5_filter + tail_hits5_filter) / 2, (head_hits10_filter + tail_hits10_filter) / 2))
        print('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format((head_precision1_filter + tail_precision1_filter) / 2, (head_precision2_filter + tail_precision2_filter) / 2, (head_precision3_filter + tail_precision3_filter) / 2, (head_precision4_filter + tail_precision4_filter) / 2, (head_precision5_filter + tail_precision5_filter) / 2, (head_precision10_filter + tail_precision10_filter) / 2))
        print('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format((head_recall1_filter + tail_recall1_filter) / 2, (head_recall2_filter + tail_recall2_filter) / 2, (head_recall3_filter + tail_recall3_filter) / 2, (head_recall4_filter + tail_recall4_filter) / 2, (head_recall5_filter + tail_recall5_filter) / 2, (head_recall10_filter + tail_recall10_filter) / 2))
        print('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format((head_f11_filter + tail_f11_filter) / 2, (head_f12_filter + tail_f12_filter) / 2, (head_f13_filter + tail_f13_filter) / 2, (head_f14_filter + tail_f14_filter) / 2, (head_f15_filter + tail_f15_filter) / 2, (head_f110_filter + tail_f110_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')
        file_object = open(self.log_file, 'a')
        file_object.write('-----Head prediction-----\r\n')
        file_object.write('MeanRank: {:.3f}'.format(head_meanrank_filter) + '\r\n')
        file_object.write('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format(head_hits1_filter, head_hits2_filter, head_hits3_filter, head_hits4_filter, head_hits5_filter, head_hits10_filter) + '\r\n')
        file_object.write('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format(head_precision1_filter, head_precision2_filter, head_precision3_filter, head_precision4_filter, head_precision5_filter, head_precision10_filter) + '\r\n')
        file_object.write('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format(head_recall1_filter, head_recall2_filter, head_recall3_filter, head_recall4_filter, head_recall5_filter, head_recall10_filter) + '\r\n')
        file_object.write('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format(head_f11_filter, head_f12_filter, head_f13_filter, head_f14_filter, head_f15_filter, head_f110_filter) + '\r\n')
        file_object.write('-----Tail prediction-----\r\n')
        file_object.write('MeanRank: {:.3f}'.format(tail_meanrank_filter) + '\r\n')
        file_object.write('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format(tail_hits1_filter, tail_hits2_filter, tail_hits3_filter, tail_hits4_filter, tail_hits5_filter, tail_hits10_filter) + '\r\n')
        file_object.write('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format(tail_precision1_filter, tail_precision2_filter, tail_precision3_filter, tail_precision4_filter, tail_precision5_filter, tail_precision10_filter) + '\r\n')
        file_object.write('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format(tail_recall1_filter, tail_recall2_filter, tail_recall3_filter, tail_recall4_filter, tail_recall5_filter, tail_recall10_filter) + '\r\n')
        file_object.write('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format(tail_f11_filter, tail_f12_filter, tail_f13_filter, tail_f14_filter, tail_f15_filter, tail_f110_filter) + '\r\n')
        file_object.write('------Average------\r\n')
        file_object.write('MeanRank: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2) + '\r\n')
        file_object.write('Hits@1: {:.8f}, Hits@2: {:.8f}, Hits@3: {:.8f}, Hits@4: {:.8f}, Hits@5: {:.8f}, Hits@10: {:.8f}' \
            .format((head_hits1_filter + tail_hits1_filter) / 2, (head_hits2_filter + tail_hits2_filter) / 2, (head_hits3_filter + tail_hits3_filter) / 2, (head_hits4_filter + tail_hits4_filter) / 2, (head_hits5_filter + tail_hits5_filter) / 2, (head_hits10_filter + tail_hits10_filter) / 2) + '\r\n')
        file_object.write('Precision@1: {:.8f}, Precision@2: {:.8f}, Precision@3: {:.8f}, Precision@4: {:.8f}, Precision@5: {:.8f}, Precision@10: {:.8f}' \
            .format((head_precision1_filter + tail_precision1_filter) / 2, (head_precision2_filter + tail_precision2_filter) / 2, (head_precision3_filter + tail_precision3_filter) / 2, (head_precision4_filter + tail_precision4_filter) / 2, (head_precision5_filter + tail_precision5_filter) / 2, (head_precision10_filter + tail_precision10_filter) / 2) + '\r\n')
        file_object.write('Recall@1: {:.8f}, Recall@2: {:.8f}, Recall@3: {:.8f}, Recall@4: {:.8f}, Recall@5: {:.8f}, Recall@10: {:.8f}' \
            .format((head_recall1_filter + tail_recall1_filter) / 2, (head_recall2_filter + tail_recall2_filter) / 2, (head_recall3_filter + tail_recall3_filter) / 2, (head_recall4_filter + tail_recall4_filter) / 2, (head_recall5_filter + tail_recall5_filter) / 2, (head_recall10_filter + tail_recall10_filter) / 2) + '\r\n')
        file_object.write('F1@1: {:.8f}, F1@2: {:.8f}, F1@3: {:.8f}, F1@4: {:.8f}, F1@5: {:.8f}, F1@10: {:.8f}' \
            .format((head_f11_filter + tail_f11_filter) / 2, (head_f12_filter + tail_f12_filter) / 2, (head_f13_filter + tail_f13_filter) / 2, (head_f14_filter + tail_f14_filter) / 2, (head_f15_filter + tail_f15_filter) / 2, (head_f110_filter + tail_f110_filter) / 2) + '\r\n')
        file_object.write('cost time: {:.3f}s'.format(timeit.default_timer() - start) + '\r\n')
        file_object.write('-----Finish evaluation-----\r\n')
        file_object.close()

    def generate_evaluation_batch(self, in_queue, out_queue):
        while True:
            raw_eval_batch = in_queue.get()
            if raw_eval_batch is None:
                return
            else:
                for head, tail, relation, graph in raw_eval_batch:
                    current_triple = (head, tail, relation, graph)
                    '''Raw'''
                    head_prediction_batch_raw = [current_triple]
                    head_neg_pool = set(self.dataset.entity_dict.values())
                    head_neg_pool.remove(head)
                    head_prediction_batch_raw.extend([(head_neg, tail, relation, graph) for head_neg in head_neg_pool])
                    tail_prediction_batch_raw = [current_triple]
                    tail_neg_pool = set(self.dataset.entity_dict.values())
                    tail_neg_pool.remove(tail)
                    tail_prediction_batch_raw.extend([(head, tail_neg, relation, graph) for tail_neg in tail_neg_pool])
                    '''Filter'''
                    head_prediction_batch_filter = [current_triple]
                    for triple_neg in head_prediction_batch_raw:
                        if triple_neg not in self.dataset.golden_triple_pool:
                            head_prediction_batch_filter.append(triple_neg)
                    tail_prediction_batch_filter = [current_triple]
                    for triple_neg in tail_prediction_batch_raw:
                        if triple_neg not in self.dataset.golden_triple_pool:
                            tail_prediction_batch_filter.append(triple_neg)
                    out_queue.put((head, tail, relation, graph, head_prediction_batch_raw, tail_prediction_batch_raw,
                                   head_prediction_batch_filter, tail_prediction_batch_filter))

    def calculate_rank(self, in_queue, out_queue):
        while True:
            eval_batch_and_score = in_queue.get()
            if eval_batch_and_score is None:
                in_queue.task_done()
                return
            else:
                head_prediction, tail_prediction, relation_prediction, graph_prediction, \
                    head_prediction_score_raw, tail_prediction_score_raw, \
                    head_prediction_score_filter, tail_prediction_score_filter = eval_batch_and_score

                head, tail, relation, graph = head_prediction.tolist(), tail_prediction.tolist(), relation_prediction.tolist(), graph_prediction.tolist()

                '''Head Raw'''
                head_prediction_rank_raw = np.argsort(head_prediction_score_raw)
                head_rank_raw = head_prediction_rank_raw.argmin()
                head_hits1_raw = 1 if head_rank_raw < 1 else 0
                head_hits2_raw = 1 if head_rank_raw < 2 else 0
                head_hits3_raw = 1 if head_rank_raw < 3 else 0
                head_hits4_raw = 1 if head_rank_raw < 4 else 0
                head_hits5_raw = 1 if head_rank_raw < 5 else 0
                head_hits10_raw = 1 if head_rank_raw < 10 else 0

                head_r1_raw = 1
                head_r2_raw = 2
                head_r3_raw = 3
                head_r4_raw = 4
                head_r5_raw = 5
                head_r10_raw = 10

                head_t1_raw = 1
                head_t2_raw = 1
                head_t3_raw = 1
                head_t4_raw = 1
                head_t5_raw = 1
                head_t10_raw = 1
                head_prediction_batch_raw = [(head, tail, relation, graph)]
                head_neg_pool = set(self.dataset.entity_dict.values())
                head_neg_pool.remove(head)
                for head_neg in head_neg_pool:
                    head_prediction_batch_raw.append((head_neg, tail, relation, graph))
                    if (head_neg, tail, relation, graph) in self.dataset.golden_triple_pool:
                        head_t1_raw += 1
                        head_t2_raw += 1
                        head_t3_raw += 1
                        head_t4_raw += 1
                        head_t5_raw += 1
                        head_t10_raw += 1

                head_rt1_raw = 0
                head_rt2_raw = 0
                head_rt3_raw = 0
                head_rt4_raw = 0
                head_rt5_raw = 0
                head_rt10_raw = 0
                if head_prediction_batch_raw[head_prediction_rank_raw[0]] in self.dataset.golden_triple_pool:
                    head_rt1_raw += 1
                    head_rt2_raw += 1
                    head_rt3_raw += 1
                    head_rt4_raw += 1
                    head_rt5_raw += 1
                    head_rt10_raw += 1
                if head_prediction_batch_raw[head_prediction_rank_raw[1]] in self.dataset.golden_triple_pool:
                    head_rt2_raw += 1
                    head_rt3_raw += 1
                    head_rt4_raw += 1
                    head_rt5_raw += 1
                    head_rt10_raw += 1
                if head_prediction_batch_raw[head_prediction_rank_raw[2]] in self.dataset.golden_triple_pool:
                    head_rt3_raw += 1
                    head_rt4_raw += 1
                    head_rt5_raw += 1
                    head_rt10_raw += 1
                if head_prediction_batch_raw[head_prediction_rank_raw[3]] in self.dataset.golden_triple_pool:
                    head_rt4_raw += 1
                    head_rt5_raw += 1
                    head_rt10_raw += 1
                if head_prediction_batch_raw[head_prediction_rank_raw[4]] in self.dataset.golden_triple_pool:
                    head_rt5_raw += 1
                    head_rt10_raw += 1
                for i in list(range(5, 10)):
                    if head_prediction_batch_raw[head_prediction_rank_raw[i]] in self.dataset.golden_triple_pool:
                        head_rt10_raw += 1

                '''Tail Raw'''
                tail_prediction_rank_raw = np.argsort(tail_prediction_score_raw)
                tail_rank_raw = tail_prediction_rank_raw.argmin()
                tail_hits1_raw = 1 if tail_rank_raw < 1 else 0
                tail_hits2_raw = 1 if tail_rank_raw < 2 else 0
                tail_hits3_raw = 1 if tail_rank_raw < 3 else 0
                tail_hits4_raw = 1 if tail_rank_raw < 4 else 0
                tail_hits5_raw = 1 if tail_rank_raw < 5 else 0
                tail_hits10_raw = 1 if tail_rank_raw < 10 else 0

                tail_r1_raw = 1
                tail_r2_raw = 2
                tail_r3_raw = 3
                tail_r4_raw = 4
                tail_r5_raw = 5
                tail_r10_raw = 10

                tail_t1_raw = 1
                tail_t2_raw = 1
                tail_t3_raw = 1
                tail_t4_raw = 1
                tail_t5_raw = 1
                tail_t10_raw = 1
                tail_prediction_batch_raw = [(head, tail, relation, graph)]
                tail_neg_pool = set(self.dataset.entity_dict.values())
                tail_neg_pool.remove(tail)
                for tail_neg in tail_neg_pool:
                    tail_prediction_batch_raw.append((head, tail_neg, relation, graph))
                    if (head, tail_neg, relation, graph) in self.dataset.golden_triple_pool:
                        tail_t1_raw += 1
                        tail_t2_raw += 1
                        tail_t3_raw += 1
                        tail_t4_raw += 1
                        tail_t5_raw += 1
                        tail_t10_raw += 1

                tail_rt1_raw = 0
                tail_rt2_raw = 0
                tail_rt3_raw = 0
                tail_rt4_raw = 0
                tail_rt5_raw = 0
                tail_rt10_raw = 0
                if tail_prediction_batch_raw[tail_prediction_rank_raw[0]] in self.dataset.golden_triple_pool:
                    tail_rt1_raw += 1
                    tail_rt2_raw += 1
                    tail_rt3_raw += 1
                    tail_rt4_raw += 1
                    tail_rt5_raw += 1
                    tail_rt10_raw += 1
                if tail_prediction_batch_raw[tail_prediction_rank_raw[1]] in self.dataset.golden_triple_pool:
                    tail_rt2_raw += 1
                    tail_rt3_raw += 1
                    tail_rt4_raw += 1
                    tail_rt5_raw += 1
                    tail_rt10_raw += 1
                if tail_prediction_batch_raw[tail_prediction_rank_raw[2]] in self.dataset.golden_triple_pool:
                    tail_rt3_raw += 1
                    tail_rt4_raw += 1
                    tail_rt5_raw += 1
                    tail_rt10_raw += 1
                if tail_prediction_batch_raw[tail_prediction_rank_raw[3]] in self.dataset.golden_triple_pool:
                    tail_rt4_raw += 1
                    tail_rt5_raw += 1
                    tail_rt10_raw += 1
                if tail_prediction_batch_raw[tail_prediction_rank_raw[4]] in self.dataset.golden_triple_pool:
                    tail_rt5_raw += 1
                    tail_rt10_raw += 1
                for i in list(range(5, 10)):
                    if tail_prediction_batch_raw[tail_prediction_rank_raw[i]] in self.dataset.golden_triple_pool:
                        tail_rt10_raw += 1

                '''Head Filter'''
                head_prediction_rank_filter = np.argsort(head_prediction_score_filter)
                head_rank_filter = head_prediction_rank_filter.argmin()
                head_hits1_filter = 1 if head_rank_filter < 1 else 0
                head_hits2_filter = 1 if head_rank_filter < 2 else 0
                head_hits3_filter = 1 if head_rank_filter < 3 else 0
                head_hits4_filter = 1 if head_rank_filter < 4 else 0
                head_hits5_filter = 1 if head_rank_filter < 5 else 0
                head_hits10_filter = 1 if head_rank_filter < 10 else 0

                head_r1_filter = 1
                head_r2_filter = 2
                head_r3_filter = 3
                head_r4_filter = 4
                head_r5_filter = 5
                head_r10_filter = 10

                head_t1_filter = 1
                head_t2_filter = 1
                head_t3_filter = 1
                head_t4_filter = 1
                head_t5_filter = 1
                head_t10_filter = 1
                head_prediction_batch_filter = [(head, tail, relation, graph)]
                for triple_neg in head_prediction_batch_raw:
                    if triple_neg not in self.dataset.golden_triple_pool:
                        head_prediction_batch_filter.append(triple_neg)
                        if triple_neg in self.dataset.golden_triple_pool:
                            head_t1_filter += 1
                            head_t2_filter += 1
                            head_t3_filter += 1
                            head_t4_filter += 1
                            head_t5_filter += 1
                            head_t10_filter += 1

                head_rt1_filter = 0
                head_rt2_filter = 0
                head_rt3_filter = 0
                head_rt4_filter = 0
                head_rt5_filter = 0
                head_rt10_filter = 0
                if head_prediction_batch_filter[head_prediction_rank_filter[0]] in self.dataset.golden_triple_pool:
                    head_rt1_filter += 1
                    head_rt2_filter += 1
                    head_rt3_filter += 1
                    head_rt4_filter += 1
                    head_rt5_filter += 1
                    head_rt10_filter += 1
                if head_prediction_batch_filter[head_prediction_rank_filter[1]] in self.dataset.golden_triple_pool:
                    head_rt2_filter += 1
                    head_rt3_filter += 1
                    head_rt4_filter += 1
                    head_rt5_filter += 1
                    head_rt10_filter += 1
                if head_prediction_batch_filter[head_prediction_rank_filter[2]] in self.dataset.golden_triple_pool:
                    head_rt3_filter += 1
                    head_rt4_filter += 1
                    head_rt5_filter += 1
                    head_rt10_filter += 1
                if head_prediction_batch_filter[head_prediction_rank_filter[3]] in self.dataset.golden_triple_pool:
                    head_rt4_filter += 1
                    head_rt5_filter += 1
                    head_rt10_filter += 1
                if head_prediction_batch_filter[head_prediction_rank_filter[4]] in self.dataset.golden_triple_pool:
                    head_rt5_filter += 1
                    head_rt10_filter += 1
                for i in list(range(5, 10)):
                    if head_prediction_batch_filter[head_prediction_rank_filter[i]] in self.dataset.golden_triple_pool:
                        head_rt10_filter += 1

                '''Tail Filter'''
                tail_prediction_rank_filter = np.argsort(tail_prediction_score_filter)
                tail_rank_filter = tail_prediction_rank_filter.argmin()
                tail_hits1_filter = 1 if tail_rank_filter < 1 else 0
                tail_hits2_filter = 1 if tail_rank_filter < 2 else 0
                tail_hits3_filter = 1 if tail_rank_filter < 3 else 0
                tail_hits4_filter = 1 if tail_rank_filter < 4 else 0
                tail_hits5_filter = 1 if tail_rank_filter < 5 else 0
                tail_hits10_filter = 1 if tail_rank_filter < 10 else 0

                tail_r1_filter = 1
                tail_r2_filter = 2
                tail_r3_filter = 3
                tail_r4_filter = 4
                tail_r5_filter = 5
                tail_r10_filter = 10

                tail_t1_filter = 1
                tail_t2_filter = 1
                tail_t3_filter = 1
                tail_t4_filter = 1
                tail_t5_filter = 1
                tail_t10_filter = 1
                tail_prediction_batch_filter = [(head, tail, relation, graph)]
                for triple_neg in tail_prediction_batch_raw:
                    if triple_neg not in self.dataset.golden_triple_pool:
                        tail_prediction_batch_filter.append(triple_neg)
                        if triple_neg in self.dataset.golden_triple_pool:
                            tail_t1_filter += 1
                            tail_t2_filter += 1
                            tail_t3_filter += 1
                            tail_t4_filter += 1
                            tail_t5_filter += 1
                            tail_t10_filter += 1

                tail_rt1_filter = 0
                tail_rt2_filter = 0
                tail_rt3_filter = 0
                tail_rt4_filter = 0
                tail_rt5_filter = 0
                tail_rt10_filter = 0
                if tail_prediction_batch_filter[tail_prediction_rank_filter[0]] in self.dataset.golden_triple_pool:
                    tail_rt1_filter += 1
                    tail_rt2_filter += 1
                    tail_rt3_filter += 1
                    tail_rt4_filter += 1
                    tail_rt5_filter += 1
                    tail_rt10_filter += 1
                if tail_prediction_batch_filter[tail_prediction_rank_filter[1]] in self.dataset.golden_triple_pool:
                    tail_rt2_filter += 1
                    tail_rt3_filter += 1
                    tail_rt4_filter += 1
                    tail_rt5_filter += 1
                    tail_rt10_filter += 1
                if tail_prediction_batch_filter[tail_prediction_rank_filter[2]] in self.dataset.golden_triple_pool:
                    tail_rt3_filter += 1
                    tail_rt4_filter += 1
                    tail_rt5_filter += 1
                    tail_rt10_filter += 1
                if tail_prediction_batch_filter[tail_prediction_rank_filter[3]] in self.dataset.golden_triple_pool:
                    tail_rt4_filter += 1
                    tail_rt5_filter += 1
                    tail_rt10_filter += 1
                if tail_prediction_batch_filter[tail_prediction_rank_filter[4]] in self.dataset.golden_triple_pool:
                    tail_rt5_filter += 1
                    tail_rt10_filter += 1
                for i in list(range(5, 10)):
                    if tail_prediction_batch_filter[tail_prediction_rank_filter[i]] in self.dataset.golden_triple_pool:
                        tail_rt10_filter += 1

                out_queue.put((head_rank_raw, head_hits1_raw, head_hits2_raw, head_hits3_raw, head_hits4_raw, head_hits5_raw, head_hits10_raw,
                               head_r1_raw, head_r2_raw, head_r3_raw, head_r4_raw, head_r5_raw, head_r10_raw,
                               head_t1_raw, head_t2_raw, head_t3_raw, head_t4_raw, head_t5_raw, head_t10_raw,
                               head_rt1_raw, head_rt2_raw, head_rt3_raw, head_rt4_raw, head_rt5_raw, head_rt10_raw,
                               tail_rank_raw, tail_hits1_raw, tail_hits2_raw, tail_hits3_raw, tail_hits4_raw, tail_hits5_raw, tail_hits10_raw,
                               tail_r1_raw, tail_r2_raw, tail_r3_raw, tail_r4_raw, tail_r5_raw, tail_r10_raw,
                               tail_t1_raw, tail_t2_raw, tail_t3_raw, tail_t4_raw, tail_t5_raw, tail_t10_raw,
                               tail_rt1_raw, tail_rt2_raw, tail_rt3_raw, tail_rt4_raw, tail_rt5_raw, tail_rt10_raw,
                               head_rank_filter, head_hits1_filter, head_hits2_filter, head_hits3_filter, head_hits4_filter, head_hits5_filter, head_hits10_filter,
                               head_r1_filter, head_r2_filter, head_r3_filter, head_r4_filter, head_r5_filter, head_r10_filter,
                               head_t1_filter, head_t2_filter, head_t3_filter, head_t4_filter, head_t5_filter, head_t10_filter,
                               head_rt1_filter, head_rt2_filter, head_rt3_filter, head_rt4_filter, head_rt5_filter, head_rt10_filter,
                               tail_rank_filter, tail_hits1_filter, tail_hits2_filter, tail_hits3_filter, tail_hits4_filter, tail_hits5_filter, tail_hits10_filter,
                               tail_r1_filter, tail_r2_filter, tail_r3_filter, tail_r4_filter, tail_r5_filter, tail_r10_filter,
                               tail_t1_filter, tail_t2_filter, tail_t3_filter, tail_t4_filter, tail_t5_filter, tail_t10_filter,
                               tail_rt1_filter, tail_rt2_filter, tail_rt3_filter, tail_rt4_filter, tail_rt5_filter, tail_rt10_filter))
                in_queue.task_done()
