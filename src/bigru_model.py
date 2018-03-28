import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.seq2seq import *
from tensorflow.contrib.framework import *
from attention_wrapper3 import *

import data_util

emb_init = tf.contrib.layers.xavier_initializer() #tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
fc_layer = tf.contrib.layers.fully_connected

class BiGRUModel(object):

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 entity_vocab_size, # entity
                 buckets,
                 state_size,
                 num_layers,
                 embedding_size,
                 max_gradient,
                 batch_size,
                 learning_rate,
                 forward_only=False,
                 dtype=tf.float32):

        entity_encode = 'cnn'
        highway = True
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.entity_vocab_size = entity_vocab_size # entity
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.state_size = state_size

        self.encoder_inputs = tf.placeholder(
            tf.int32, shape=[self.batch_size, None], name='1')
        self.decoder_inputs = tf.placeholder(
            tf.int32, shape=[self.batch_size, None], name='2')
        self.decoder_targets = tf.placeholder(
            tf.int32, shape=[self.batch_size, None], name='3')
        self.encoder_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='4')
        self.decoder_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='5')
        self.beam_tok = tf.placeholder(tf.int32, shape=[self.batch_size], name='6')
        self.prev_att = tf.placeholder(
            tf.float32, shape=[self.batch_size, state_size * 2], name='7')
        self.K = tf.placeholder(tf.int32)
        self.lvt_dict = tf.placeholder(tf.int32, shape=[None], name='8')
        self.lvt_len = tf.placeholder(tf.int32, name='9')
        self.batch_dec_len = tf.placeholder(tf.int32, name='10')

        # entity
        self.entity_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None])
        self.entity_len = tf.placeholder(tf.int32, shape=[self.batch_size])
        
        encoder_fw_cells = []
        encoder_bw_cells = []
        for _ in range(2):
            encoder_fw_cells.append(tf.contrib.rnn.GRUCell(state_size))
            encoder_bw_cells.append(tf.contrib.rnn.GRUCell(state_size))

        if not forward_only:
            for i in range(2):
                encoder_fw_cells[i] = tf.contrib.rnn.DropoutWrapper(
                    encoder_fw_cells[i], output_keep_prob=0.50)
                encoder_bw_cells[i] = tf.contrib.rnn.DropoutWrapper(
                    encoder_bw_cells[i], output_keep_prob=0.50)
        encoder_fw_cell = tf.contrib.rnn.MultiRNNCell(encoder_fw_cells)
        encoder_bw_cell = tf.contrib.rnn.MultiRNNCell(encoder_bw_cells)
        #decode
        decoder_cells = []
        for _ in range(2):
            decoder_cells.append(tf.contrib.rnn.GRUCell(state_size))
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

        self.loss = tf.constant(0)
        
        with tf.variable_scope("seq2seq", dtype=dtype):
            with tf.variable_scope("encoder"):

                self.encoder_emb = tf.get_variable(
                    "embedding", [source_vocab_size, embedding_size],
                    initializer=emb_init)

                encoder_inputs_emb = tf.nn.embedding_lookup(
                    self.encoder_emb, self.encoder_inputs)
                
                encoder_fw_cell = tf.contrib.rnn.MultiRNNCell(encoder_fw_cells)
                encoder_bw_cell = tf.contrib.rnn.MultiRNNCell(encoder_bw_cells)
                
                encoder_outputs, encoder_states = \
                    tf.nn.bidirectional_dynamic_rnn(
                        encoder_fw_cell, encoder_bw_cell, encoder_inputs_emb,
                        sequence_length=self.encoder_len, dtype=dtype)
                
                encoder_len = self.encoder_len

                if forward_only:
                   encoder_outputs = tile_batch(encoder_outputs, multiplier=10)
                   encoder_states = nest.map_structure(lambda s: tile_batch(s, 10), encoder_states)
                   encoder_len = tile_batch(self.encoder_len, multiplier=10)
                
                #encoder_states = encoder_states[-1]
            
            
            if entity_encode == 'no':
                # NO
                with tf.variable_scope("entity_encoder"):
                    self.entity_emb = tf.get_variable(
                        "embedding", [entity_vocab_size, 1000],
                        initializer=emb_init)
                    
                    entity_vector = tf.nn.embedding_lookup(
                        self.entity_emb, self.entity_inputs)
            
            elif entity_encode == 'rnn':
                # RNN
                with tf.variable_scope("entity_encoder"):
                    entity_fw_cell = tf.contrib.rnn.GRUCell(state_size)
                    entity_bw_cell = tf.contrib.rnn.GRUCell(state_size)
                    
                    if not forward_only:
                        entity_fw_cell = tf.contrib.rnn.DropoutWrapper(entity_fw_cell, output_keep_prob=0.5)
                        entity_bw_cell = tf.contrib.rnn.DropoutWrapper(entity_bw_cell, output_keep_prob=0.5)
                    
                    self.entity_emb = tf.get_variable(
                        "embedding", [entity_vocab_size, 1000],
                        initializer=emb_init)
                    
                    entity_inputs_emb = tf.nn.embedding_lookup(
                        self.entity_emb, self.entity_inputs)
                   
                    entity_outputs, entity_states = \
                        tf.nn.bidirectional_dynamic_rnn(
                            entity_fw_cell, entity_bw_cell, entity_inputs_emb,
                            sequence_length=self.entity_len, dtype=dtype)
                    
                    entity_vector = tf.concat(entity_outputs, 2)
                    entity_vector.set_shape([self.batch_size, None, state_size*2])
                    entity_proj = entity_inputs_emb
            
            elif entity_encode == 'cnn':
                # CNN
                with tf.variable_scope("entity_encoder"):
                    self.entity_emb = tf.get_variable(
                        "embedding", [entity_vocab_size, 1000],
                        initializer=emb_init)
                        
                    entity_inputs_emb = tf.nn.embedding_lookup(
                        self.entity_emb, self.entity_inputs)
                    
                    entity_inputs_emb_expanded = tf.expand_dims(entity_inputs_emb, -1)
                    filter_sizes = [3,5,7]
                    num_filters = [400,300,300]
                    
                    outputs = []
                    for i, filter_size in enumerate(filter_sizes):
                        with tf.name_scope("conv-maxpool-%s" % filter_size):
                            filter_shape = [filter_size, 1000, 1, num_filters[i]]
                            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                            b = tf.Variable(tf.constant(0.1, shape=[num_filters[i]]), name="b")
                            conv = tf.nn.conv2d(
                                entity_inputs_emb_expanded,
                                W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                            erase = int((7 - filter_size) / 2)
                            if erase != 0:
                                h = h[:,erase:-erase,:,:]
                            if not forward_only:
                                h = tf.nn.dropout(h, 0.5)
                            outputs.append(h)
                    
                    entity_vector = tf.concat(outputs, axis=3)
                    entity_vector = tf.squeeze(entity_vector, 2)
                    entity_vector.set_shape([self.batch_size, None, state_size*2])
                    entity_proj = entity_inputs_emb[:,3:-3,:]
            
            if highway:
                # y
#                entity_proj = entity_inputs_emb[:,3:-3,:]
                Wh = tf.get_variable("Wh", [1000, 1000], initializer=xavier_initializer())
                bh = tf.Variable(tf.constant(0.0, shape=[1000]))
                entity_proj = tf.nn.tanh(tf.tensordot(entity_proj, Wh, 1) + bh)
                if not forward_only:
                    entity_proj.set_shape([self.batch_size, None, 1000])
                else:
                    entity_proj.set_shape([self.batch_size*10, None, 1000])
                
                if not forward_only:
                    entity_proj = tf.nn.dropout(entity_proj, keep_prob=0.5)
                
                # t
                Wt = tf.get_variable("Wt", [1000, 1], initializer=xavier_initializer())
                bt = tf.Variable(tf.constant(0.0, shape=[1]))
                t = tf.nn.sigmoid(tf.tensordot(entity_vector, Wt, 1) + bt)
                if not forward_only:
                    t.set_shape([self.batch_size, None, 1000])
                else:
                    t.set_shape([self.batch_size*10, None, 1000])
                self.t = t
                
                entity_vector = t*entity_vector + (1-t)*entity_proj
            
            with tf.variable_scope("init_state"):
                init_states = []
                for i in range(2):
                    init_state = fc_layer(tf.concat(encoder_states[i], 1), state_size)
                    init_states.append(init_state)
                # the shape of bidirectional_dynamic_rnn is weird
                # None for batch_size
                self.init_states = init_states
                #self.init_state.set_shape([self.batch_size, state_size])
                self.att_states = tf.concat(encoder_outputs, 2)
                    
            #with tf.variable_scope("entity_init_state"):
            #    entity_init_state = fc_layer(
            #        tf.concat(entity_states, 1), state_size)
            #    self.entity_init_state = entity_init_state
            #    self.entity_init_state.set_shape([self.batch_size, state_size])
            #    self.entity_att_states = tf.concat(entity_outputs, 2)
            #    self.entity_att_states.set_shape([self.batch_size, None, state_size*2])
            
            with tf.variable_scope("entity_attention"):
                X = tf.get_variable("X", shape=[1000, state_size], initializer=xavier_initializer())
                x = tf.get_variable("x", shape=[state_size], initializer=xavier_initializer())
                Y = tf.get_variable("Y", shape=[state_size*2, state_size], initializer=xavier_initializer())
                first = tf.matmul(tf.concat(encoder_states[-1], 1), Y)
                first = tf.expand_dims(first, 1)
                other = tf.tensordot(entity_vector, X, 1)
                weights = tf.nn.tanh(first + other)
                if not forward_only:
                    weights = tf.nn.dropout(weights, keep_prob=0.5)
                weights = tf.tensordot(weights, x, 1)
                if not forward_only:
                    weights.set_shape([self.batch_size, None])
                else:
                    weights.set_shape([10*self.batch_size, None])
                
                
                k_values, k_indices = tf.nn.top_k(weights, k=self.K)
                my_range = tf.expand_dims(tf.range(0, k_indices.shape[0]), 1)
                #print(my_range)
                my_range_repeated = tf.tile(my_range, [1, self.K])
                
                full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(k_indices, 2)], 2)
                full_indices = tf.reshape(full_indices, [-1, 2])
                
                output_shape = tf.shape(weights)
                zeros = tf.sparse_to_dense(full_indices, output_shape, 0.0, default_value=-1000000000.0, validate_indices=False)
                
                weights = tf.nn.softmax(zeros+weights)
                weights = tf.expand_dims(weights, -1)
                self.weights = weights
                context = tf.multiply(entity_vector, weights)
                context = tf.reduce_sum(context, axis=1)

            with tf.variable_scope("attention"):
                attention = BahdanauAttention(
                    state_size, self.att_states, encoder_len)

            with tf.variable_scope("decoder") as scope:
                #decoder_cells = []
                #for _ in range(2):
                #    decoder_cells.append(tf.contrib.rnn.GRUCell(state_size))
                
                if not forward_only:
                    for i in range(2):
                        decoder_cells[i] = tf.contrib.rnn.DropoutWrapper(
                            decoder_cells[i], output_keep_prob=0.50)
                
                #for i in range(2):
                decoder_cells[-1] = AttentionWrapper(
                    decoder_cells[-1], attention, state_size, context=context)
            
                initial_states = [state for state in init_states]
                if not forward_only:
                    initial_states[-1] = decoder_cells[-1].zero_state(batch_size=self.batch_size, dtype=tf.float32)
                else:
                    initial_states[-1] = decoder_cells[-1].zero_state(batch_size=10*self.batch_size, dtype=tf.float32)
                
                decoder_initial_state = tuple(initial_states)
                
                decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)
                
                self.decoder_emb = tf.get_variable(
                    "embedding", [target_vocab_size, embedding_size],
                    initializer=emb_init)
                output_layer = tf.contrib.keras.layers.Dense(target_vocab_size, name="train_output")
                if not forward_only:
                    #output_layer = tf.contrib.keras.layers.Dense(target_vocab_size, name="train_output")
                    
                    decoder_inputs_emb = tf.nn.embedding_lookup(
                        self.decoder_emb, self.decoder_inputs)
                    helper = tf.contrib.seq2seq.TrainingHelper(
                        decoder_inputs_emb, self.decoder_len)
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        decoder_cell, helper, decoder_initial_state, output_layer)
                    
                    outputs, final_state, _ = \
                        tf.contrib.seq2seq.dynamic_decode(decoder)
                    
                    outputs_logits = tf.identity(outputs.rnn_output)
                    self.outputs = outputs_logits
                    
                    weights = tf.sequence_mask(
                        self.decoder_len, dtype=tf.float32)

                    self.loss_t = weights
                    loss_t = tf.contrib.seq2seq.sequence_loss(
                        outputs_logits, self.decoder_targets, weights,
                        average_across_timesteps=False,
                        average_across_batch=False)
                    self.loss = tf.reduce_sum(loss_t) / self.batch_size

                    params = tf.trainable_variables()
                    opt = tf.train.AdadeltaOptimizer(
                        self.learning_rate, epsilon=1e-6)
                    gradients = tf.gradients(self.loss, params)
                    clipped_gradients, norm = \
                        tf.clip_by_global_norm(gradients, max_gradient)
                    self.updates = opt.apply_gradients(
                        zip(clipped_gradients, params),
                        global_step=self.global_step)

                    tf.summary.scalar('loss', self.loss)
                else:                    
                    #output_layer = tf.contrib.keras.layers.Dense(target_vocab_size, name="test_output", trainable=True)                    
                    st_toks = tf.convert_to_tensor(
                        [data_util.ID_GO]*self.batch_size, dtype=tf.int32)
                    
                    def embed_proj(inputs):
                        return tf.nn.embedding_lookup(self.decoder_emb, inputs)
                    
                
                    #decoding_helper = GreedyEmbeddingHelper(start_tokens=st_toks, end_token=data_util.ID_EOS, embedding=embed_and_input_proj)
                    inference_decoder = BeamSearchDecoder(cell=decoder_cell, embedding=embed_proj, start_tokens=st_toks, end_token=data_util.ID_EOS, initial_state=decoder_initial_state, beam_width=10, output_layer=output_layer)
                    
                    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, output_time_major=False, maximum_iterations=100)
                    self.outputs = outputs.predicted_ids[:,:,0]
                    #self.outputs = tf.transpose(outputs.predicted_ids, [0,2,1])
                    print(self.outputs)
        
        self.saver = tf.train.Saver(tf.global_variables())
        #self.saver = tf.train.Saver()
        self.summary_merge = tf.summary.merge_all()
    
    def step(self,
             session,
             encoder_inputs,
             decoder_inputs,
             entity_inputs,
             encoder_len,
             decoder_len,
             entity_len,
             K,
             forward_only,
             summary_writer=None):

        # dim fit is important for sequence_mask
        # TODO better way to use sequence_mask
        if encoder_inputs.shape[1] != max(encoder_len):
            raise ValueError("encoder_inputs and encoder_len does not fit")
        if not forward_only and \
            decoder_inputs.shape[1] != max(decoder_len) + 1:
            raise ValueError("decoder_inputs and decoder_len does not fit")
        input_feed = {}
        input_feed[self.encoder_inputs] = encoder_inputs
        input_feed[self.entity_inputs] = entity_inputs
        input_feed[self.decoder_inputs] = decoder_inputs[:, :-1]
        input_feed[self.decoder_targets] = decoder_inputs[:, 1:]
        input_feed[self.encoder_len] = encoder_len
        input_feed[self.entity_len] = entity_len
        input_feed[self.decoder_len] = decoder_len
        input_feed[self.K] = K
        input_feed[self.prev_att] = np.zeros(
            [self.batch_size, 2 * self.state_size])

        if forward_only:
            output_feed = [self.loss, self.outputs]
        else:
            output_feed = [self.loss, self.updates]
        
        output_feed += [self.weights, self.t]

        if summary_writer:
            output_feed += [self.summary_merge, self.global_step]
        
        outputs = session.run(output_feed, input_feed)
        
        if summary_writer:
            summary_writer.add_summary(outputs[4], outputs[5])
        return outputs[:4]

    def step_beam(self,
                  session,
                  encoder_inputs,
                  encoder_len,
                  entity_inputs,
                  entity_len,
                  K,
                  max_len=12,
                  geneos=True):

        beam_size = self.batch_size

        if encoder_inputs.shape[0] == 1:
            encoder_inputs = np.repeat(encoder_inputs, beam_size, axis=0)
            encoder_len = np.repeat(encoder_len, beam_size, axis=0)

        if encoder_inputs.shape[1] != max(encoder_len):
            raise ValueError("encoder_inputs and encoder_len does not fit")
        #generate attention_states
        input_feed = {}
        input_feed[self.encoder_inputs] = encoder_inputs
        input_feed[self.encoder_len] = encoder_len
        input_feed[self.entity_inputs] = entity_inputs
        input_feed[self.entity_len] = entity_len
        input_feed[self.K] = K
        output_feed = [self.att_states, self.init_states[-1]]
        outputs = session.run(output_feed, input_feed)

        att_states = outputs[0]
        prev_state = outputs[1]
        prev_tok = np.ones([beam_size], dtype="int32") * data_util.ID_GO
        prev_att = np.zeros([self.batch_size, 2 * self.state_size])

        input_feed = {}
        input_feed[self.att_states] = att_states
        input_feed[self.encoder_inputs] = encoder_inputs
        input_feed[self.encoder_len] = encoder_len
        input_feed[self.entity_inputs] = entity_inputs
        input_feed[self.entity_len] = entity_len
        input_feed[self.K] = K

        ret = [[]] * beam_size
        neos = np.ones([beam_size], dtype="bool")

        score = np.ones([beam_size], dtype="float32") * (-1e8)
        score[0] = 0

        beam_att = np.zeros(
            [self.batch_size, self.state_size*2], dtype="float32")

        for i in range(max_len):
            input_feed[self.init_states[-1]] = prev_state
            input_feed[self.beam_tok] = prev_tok
            input_feed[self.prev_att] = beam_att
            output_feed = [self.beam_nxt_state[1],
                           self.beam_logsoftmax,
                           self.beam_nxt_state[0]]

            outputs = session.run(output_feed, input_feed)

            beam_att = outputs[0]
            print(outputs[1].shape)
            tok_logsoftmax = np.asarray(outputs[1])
            tok_logsoftmax = tok_logsoftmax.reshape(
                [beam_size, self.target_vocab_size])
            if not geneos:
                tok_logsoftmax[:, data_util.ID_EOS] = -1e8

            tok_argsort = np.argsort(tok_logsoftmax, axis=1)[:, -beam_size:]
            tmp_arg0 = np.arange(beam_size).reshape([beam_size, 1])
            tok_argsort_score = tok_logsoftmax[tmp_arg0, tok_argsort]
            tok_argsort_score *= neos.reshape([beam_size, 1])
            tok_argsort_score += score.reshape([beam_size, 1])
            all_arg = np.argsort(tok_argsort_score.flatten())[-beam_size:]
            arg0 = all_arg // beam_size #previous id in batch
            arg1 = all_arg % beam_size
            prev_tok = tok_argsort[arg0, arg1] #current word
            prev_state = outputs[2][arg0]
            score = tok_argsort_score[arg0, arg1]

            neos = neos[arg0] & (prev_tok != data_util.ID_EOS)

            ret_t = []
            for j in range(beam_size):
                ret_t.append(ret[arg0[j]] + [prev_tok[j]])

            ret = ret_t
        return ret[-1]

    def add_pad(self, data, fixlen):
        data = map(lambda x: x + [data_util.ID_PAD] * (fixlen - len(x)), data)
        data = list(data)
        return np.asarray(data)

    def get_batch(self, data, bucket_id):
        encoder_inputs, entity_inputs, decoder_inputs = [], [], []
        encoder_len, entity_len, decoder_len = [], [], []

        # Get a random batch of encoder and decoder inputs from data,
        # and add GO to decoder.
        for _ in range(self.batch_size):
            shiva = random.choice(data[bucket_id])
            #print(shiva)
            encoder_input, decoder_input, entity_input = shiva

            encoder_inputs.append(encoder_input)
            encoder_len.append(len(encoder_input))

            decoder_inputs.append(decoder_input)
            decoder_len.append(len(decoder_input))
            
            entity_inputs.append(entity_input)
            entity_len.append(len(entity_input))

        batch_enc_len = max(encoder_len)
        batch_dec_len = max(decoder_len)
        batch_ent_len = max(entity_len)

        encoder_inputs = self.add_pad(encoder_inputs, batch_enc_len)
        decoder_inputs = self.add_pad(decoder_inputs, batch_dec_len)
        entity_inputs = self.add_pad(entity_inputs, batch_ent_len)
        encoder_len = np.asarray(encoder_len)
        entity_len = np.asarray(entity_len)
        # decoder_input has both <GO> and <EOS>
        # len(decoder_input)-1 is number of steps in the decoder.
        decoder_len = np.asarray(decoder_len) - 1

        return encoder_inputs, decoder_inputs, encoder_len, decoder_len, entity_inputs, entity_len
