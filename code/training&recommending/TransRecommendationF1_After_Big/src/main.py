from dataset import Dataset
from model import TransE

import numpy as np
import tensorflow as tf
import argparse

import os

def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_dir', type=str, default='../data/after_big/')
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--margin_value', type=float, default=1.0)
    parser.add_argument('--score_func', type=str, default='L1')
    parser.add_argument('--batch_size', type=int, default=4800)
    parser.add_argument('--eval_batch_size', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_generator', type=int, default=24)
    parser.add_argument('--n_rank_calculator', type=int, default=24)
    parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
    parser.add_argument('--summary_dir', type=str, default='../summary/')
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--eval_freq', type=int, default=200)
    parser.add_argument('--log_file', type=str, default='../log/log_after_big.txt')

    args = parser.parse_args()
    print(args)

    file_object = open(args.log_file, 'w')
    file_object.close()

    fb15k = Dataset(data_dir=args.data_dir, log_file=args.log_file)
    kge_model = TransE(dataset=fb15k, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                       score_func=args.score_func, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                       learning_rate=args.learning_rate, n_generator=args.n_generator,
                       n_rank_calculator=args.n_rank_calculator, log_file=args.log_file)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    with tf.Session(config=sess_config) as sess:
        print('-----Initializing tf graph-----')
        file_object = open(args.log_file, 'a')
        file_object.write('-----Initializing tf graph-----\r\n')
        file_object.close()
        tf.global_variables_initializer().run()
        print('-----Initialization accomplished-----')
        print('----Check norm----')
        file_object = open(args.log_file, 'a')
        file_object.write('-----Initialization accomplished-----\r\n')
        file_object.write('----Check norm----\r\n')
        file_object.close()
        entity_embedding = kge_model.entity_embedding.eval(session=sess)
        relation_embedding = kge_model.relation_embedding.eval(session=sess)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))
        file_object = open(args.log_file, 'a')
        file_object.write('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm) + '\r\n')
        file_object.close()
        summary_writer = tf.summary.FileWriter(logdir=args.summary_dir, graph=sess.graph)
        saver = tf.train.Saver(max_to_keep=1)
        for epoch in range(args.max_epoch):
            print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
            file_object = open(args.log_file, 'a')
            file_object.write('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30 + '\r\n')
            file_object.close()
            kge_model.launch_training(session=sess, summary_writer=summary_writer)
            saver.save(sess, '../ckpt/after_big.ckpt', global_step=epoch + 1)
            if (epoch + 1) % args.eval_freq == 0:
                kge_model.launch_evaluation(session=sess)


if __name__ == '__main__':
    main()
