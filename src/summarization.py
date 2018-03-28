import logging
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

import bigru_model
import data_util

tf.app.flags.DEFINE_float("learning_rate", 1., "Learning rate.")
tf.app.flags.DEFINE_integer("size", 500, "Size of hidden layers.")
tf.app.flags.DEFINE_integer("embsize", 300, "Size of embedding.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("test_file", "", "Test filename.")
tf.app.flags.DEFINE_string("test_output", "output.txt", "Test output.")
tf.app.flags.DEFINE_string("train_dir", "model", "Training directory.")
tf.app.flags.DEFINE_string("tfboard", "tfboard", "Tensorboard log directory.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for testing.")
tf.app.flags.DEFINE_boolean("geneos", True, "Do not generate EOS. ")
tf.app.flags.DEFINE_float(
    "max_gradient", 1.0, "Clip gradients l2 norm to this range.")
tf.app.flags.DEFINE_integer(
    "batch_size", 10, "Batch size in training / beam size in testing.")
tf.app.flags.DEFINE_integer(
    "doc_vocab_size", 39991, "Document vocabulary size.")
tf.app.flags.DEFINE_integer(
    "sum_vocab_size", 38223, "Summary vocabulary size.")
tf.app.flags.DEFINE_integer(
    "max_train", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer(
    "max_iter", 1000000, "Maximum training iterations.")
tf.app.flags.DEFINE_integer(
    "steps_per_validation", 1000, "Training steps between validations.")
tf.app.flags.DEFINE_integer(
    "steps_per_checkpoint", 10000, "Training steps between checkpoints.")
tf.app.flags.DEFINE_string(
    "checkpoint", "", "Checkpoint to load (use up-to-date if not set)")
tf.app.flags.DEFINE_integer(
    "en_vocab_size", 25000, "Entity vocabulary size.")
tf.app.flags.DEFINE_integer(
    "K", 10, "Entity vocabulary size.")
tf.app.flags.DEFINE_integer(
    "lvt", 5000, "LVT size.")
FLAGS = tf.app.flags.FLAGS

# We use a number of buckets for sampling
#_buckets = [(30, 10), (50, 20), (70, 20), (100, 20), (200, 30)]
#_buckets = [(25, 16), (35, 8), (35, 16), (50, 8), (50, 16)]
#_buckets = [(500, 15), (600, 20), (800, 20), (1200, 20), (2000, 20)]
_buckets = [(500, 60), (600, 80), (800, 80), (1200, 80), (2000, 80)]


def create_bucket(source, target):
    data_set = [[] for _ in _buckets]
    for s, t in zip(source, target):
        t = [data_util.ID_GO] + t + [data_util.ID_EOS]
        for bucket_id, (s_size, t_size) in enumerate(_buckets):
            if len(s) <= s_size and len(t) <= t_size:
                data_set[bucket_id].append([s, t])
                break
    return data_set
#entity
def create_bucket(source, target, entity):
    data_set = [[] for _ in _buckets]
    for s, t, e in zip(source, target, entity):
        t = [data_util.ID_GO] + t + [data_util.ID_EOS]
        if len(e) == 0:
            e = [data_util.ID_PAD]
        e = [data_util.ID_PAD, data_util.ID_PAD, data_util.ID_PAD] + e + [data_util.ID_PAD, data_util.ID_PAD, data_util.ID_PAD]
        for bucket_id, (s_size, t_size) in enumerate(_buckets):
            if len(s) <= s_size and len(t) <= t_size:
                data_set[bucket_id].append([s, t, e])
                break
    return data_set


def create_model(session, forward_only, document_vecs, summary_vecs, entity_vecs):
    """Create model and initialize or load parameters in session."""
    dtype = tf.float32
    model = bigru_model.BiGRUModel(
        FLAGS.doc_vocab_size,
        FLAGS.sum_vocab_size,
        FLAGS.en_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.embsize,
        FLAGS.max_gradient,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        forward_only=forward_only,
        dtype=dtype)
    if FLAGS.checkpoint != "":
        ckpt = FLAGS.checkpoint
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt:
            ckpt = ckpt.model_checkpoint_path
    if ckpt and tf.train.checkpoint_exists(ckpt):
        logging.info("Reading model parameters from %s" % ckpt)
        model.saver.restore(session, ckpt)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        session.run(model.entity_emb.assign(entity_vecs))
        session.run(model.encoder_emb.assign(document_vecs))
        session.run(model.decoder_emb.assign(summary_vecs))
    return model
    
def train():
    with tf.Session() as sess:
        #model = create_model(sess, False, None, None, None)
        logging.info("Preparing summarization data.")
        docid, sumid, doc_dict, sum_dict, enid, en_dict, doc_vec, sum_vec, en_vec = \
            data_util.load_data(
                FLAGS.data_dir + "/train.article.txt",
                FLAGS.data_dir + "/train.title.txt",
                FLAGS.data_dir + "/doc_dict.txt",
                FLAGS.data_dir + "/sum_dict.txt",
                FLAGS.data_dir + "/train.entity.txt",
                FLAGS.data_dir + "/en_dict.txt",
                FLAGS.data_dir + "/word_vecs.txt",
                FLAGS.data_dir + "/entity_vecs.txt",
                FLAGS.doc_vocab_size, FLAGS.sum_vocab_size, FLAGS.en_vocab_size)

        val_docid, val_sumid, val_enid = \
            data_util.load_valid_data(
                FLAGS.data_dir + "/valid.article.filter.txt",
                FLAGS.data_dir + "/valid.title.filter.txt",
                doc_dict, sum_dict,
                FLAGS.data_dir + "/valid.entity.txt",
                en_dict)
        
        # Create model.
        logging.info("Creating %d layers of %d units." %
                     (FLAGS.num_layers, FLAGS.size))
        train_writer = tf.summary.FileWriter(FLAGS.tfboard, sess.graph)
        model = create_model(sess, False, doc_vec, sum_vec, en_vec)
        
        # Read data into buckets and compute their sizes.
        logging.info("Create buckets.")
        dev_set = create_bucket(val_docid, val_sumid, val_enid)
        train_set = create_bucket(docid, sumid, enid)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [
            sum(train_bucket_sizes[:i + 1]) / train_total_size
            for i in range(len(train_bucket_sizes))]

        for (s_size, t_size), nsample in zip(_buckets, train_bucket_sizes):
            logging.info("Train set bucket ({}, {}) has {} samples.".format(
                s_size, t_size, nsample))

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = sess.run(model.global_step)
        current_ppx = 100000

        while current_step <= FLAGS.max_iter:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, encoder_len, decoder_len, entity_inputs, entity_len = \
                model.get_batch(train_set, bucket_id)
            K = min(FLAGS.K, np.amax(entity_len)-6)
            step_loss, out, _, _ = model.step(
                sess, encoder_inputs, decoder_inputs, entity_inputs,
                encoder_len, decoder_len, entity_len, K, False, train_writer)
            
            #print(step_loss)
            
            step_time += (time.time() - start_time) / \
                FLAGS.steps_per_validation
            loss += step_loss * FLAGS.batch_size / np.sum(decoder_len) \
                / FLAGS.steps_per_validation
            current_step += 1

            #print(loss)
            
            # Once in a while, we save checkpoint.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)

            # Once in a while, we print statistics and run evals.
            if current_step % FLAGS.steps_per_validation == 0:
                # Print statistics for the previous epoch.
                perplexity = np.exp(float(loss))
                logging.info(
                    "global step %d step-time %.2f ppl %.2f" % (model.global_step.eval(), step_time, perplexity))

                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                total_ppx = 0
                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        logging.info("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, encoder_len, decoder_len, entity_inputs, entity_len =\
                        model.get_batch(dev_set, bucket_id)
                    K = min(FLAGS.K, np.amin(entity_len))
                    eval_loss, _, _, _ = model.step(sess, encoder_inputs,
                                            decoder_inputs, entity_inputs, encoder_len,
                                            decoder_len, entity_len, K, True)
                    eval_loss = eval_loss * FLAGS.batch_size \
                        / np.sum(decoder_len)
                    eval_ppx = np.exp(float(eval_loss))
                    total_ppx += eval_ppx
                    logging.info("  eval: bucket %d ppl %.2f" %
                                 (bucket_id, eval_ppx))
                sys.stdout.flush()
                
                if total_ppx < current_ppx:
                    current_ppx = total_ppx
                    checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                    model.saver.save(sess, checkpoint_path)

def decode():
    # Load vocabularies.
    doc_dict = data_util.load_dict(FLAGS.data_dir + "/doc_dict.txt")
    sum_dict = data_util.load_dict(FLAGS.data_dir + "/sum_dict.txt")
    en_dict = data_util.load_dict(FLAGS.data_dir + "/en_dict.txt")
    if doc_dict is None or sum_dict is None:
        logging.warning("Dict not found.")    
    data, en_data = data_util.load_test_data(FLAGS.test_file, doc_dict, FLAGS.data_dir+"/test.entity.txt", en_dict)

    with tf.Session() as sess:
        # Create model and load parameters.
        logging.info("Creating %d layers of %d units." %
                     (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, True, None, None, None)

        result = []
        for idx, token_ids in enumerate(data):
            en_ids = en_data[idx]
            if len(en_ids) == 0:
                en_ids = [data_util.ID_PAD]
#            token_ids, en_ids = d
            #print(idx)
            #print(token_ids)

            # Get a 1-element batch to feed the sentence to the model.
            shiva = model.get_batch(
                    {0: [(token_ids, [data_util.ID_GO, data_util.ID_EOS], [data_util.ID_PAD, data_util.ID_PAD, data_util.ID_PAD] + en_ids + [data_util.ID_PAD, data_util.ID_PAD, data_util.ID_PAD])]}, 0)
            #print(shiva)
            encoder_inputs, decoder_inputs, encoder_len, decoder_len, entity_inputs, entity_len = shiva
            K = min(FLAGS.K, np.amax(entity_len)-6)
            #print("K", K)

            if FLAGS.batch_size == 1 and FLAGS.geneos:
                loss, outputs, att, t = model.step(sess,
                    encoder_inputs, decoder_inputs, entity_inputs, 
                    encoder_len, decoder_len, entity_len, K, True)

                #outputs = [np.argmax(item) for item in outputs[0]]
            else:
                outputs = model.step_beam(
                    sess, encoder_inputs, encoder_len, entity_inputs, entity_len, K, geneos=FLAGS.geneos)

            # If there is an EOS symbol in outputs, cut them at that point.
            #print(outputs)
            f2 = open(FLAGS.test_output + '.disambig', 'a')
            f2.write(' '.join(str(y) + ":" + str(x.mean()) for x,y in zip(t[0], entity_inputs[0][3:])) + '\n')
            f2.close()
            f2 = open(FLAGS.test_output + '.attention', 'a')
            f2.write(' '.join(str(y) + ":" + str(x) for x,y in zip(att[0], entity_inputs[0][3:])) + '\n')
            f2.close()
            outputs = list(outputs[0])
            if data_util.ID_EOS in output:
                outputs = outputs[:outputs.index(data_util.ID_EOS)]
            #outputs = list(outputs)
            gen_sum = " ".join(data_util.sen_map2tok(outputs, sum_dict[1])) #sum_dict[1])) #lvt_str
            gen_sum = data_util.sen_postprocess(gen_sum)
            result.append(gen_sum)
            logging.info("Finish {} samples. :: {}".format(idx, gen_sum[:75]))

        with open(FLAGS.test_output, "w") as f:
            for item in result:
                print(item, file=f)

def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')
    try:
        os.makedirs(FLAGS.train_dir)
    except:
        pass
    tf.app.run()
