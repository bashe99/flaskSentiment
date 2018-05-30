import random

from collections import defaultdict
import numpy as np
import os
import time
import tensorflow as tf
from tabulate import tabulate


from .data_utils import minibatches, pad_sequences, get_CoNLL_dataset, create_memory_file_from_words,\
                        get_processing_word, merge_datasets
from .general_utils import Progbar
from .base_model import BaseModel
from .my_rnn_cell_impl import MyGRUCell, MyLSTMCell
from .metrics_calc import MetricCalc
from .attention_utils import attention


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_task = {idx: task for task, idx in self.config.vocab_tasks.items()}
        self.tasks_idx_to_intent = [{idx: intent for intent, idx in
                                     vocab.items()} for vocab in self.config.vocab_tasks_intents]
        self.tasks_idx_to_tag = [{idx: tag for tag, idx in
                                  vocab_tags.items()} for vocab_tags in self.config.vocab_tasks_tags]

        self.binary_weights_ops = [tf.no_op()]

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence, 2 * self.config.ndict_types)
        # TODO: dict_ids could be as real id, and convert to one-hot in this file
        self.dict_ids = tf.placeholder(tf.int32, shape=[None, None, 2 * self.config.ndict_types],
                        name="dict_ids")

        # shape = (batch size, max length of sentence, max length of word + 2 - 2)
        self.letter_trigram_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="letter_trigram_ids")

        # shape = (batch size)
        self.domains = tf.placeholder(tf.int32, shape=[None],
                                      name="domains")

        # shape = (batch size)
        self.intents = tf.placeholder(tf.int32, shape=[None],
                                      name="intents")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, domains=None, intents=None, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            domains: list of domain
            intents: list of intent
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        dict_ids, letter_trigram_ids, char_ids, word_ids = zip(*words)
        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        if self.config.use_chars:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)

        if self.config.use_dict:
            dict_ids, _ = pad_sequences(dict_ids, pad_tok=0, nlevels=2)

        if self.config.use_letter_trigram:
            l3t_pad_tok = 0
            if self.config.letter_trigram_dummy_row_enabled:
                l3t_pad_tok = self.config.nletter_trigrams
            letter_trigram_ids, _ = pad_sequences(letter_trigram_ids, pad_tok=l3t_pad_tok,
                nlevels=2)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if self.config.use_dict:
            feed[self.dict_ids] = dict_ids

        if self.config.use_letter_trigram:
            feed[self.letter_trigram_ids] = letter_trigram_ids

        if domains is not None:
            feed[self.domains] = domains

        if intents is not None:
            feed[self.intents] = intents

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def get_rnn_cell(self, hidden_size, state_is_tuple=True):
        """Get rnn cell according to config"""
        rnn_cell_name = self.config.my_rnn_cell.lower()
        if rnn_cell_name == "gru":
            return tf.contrib.rnn.GRUCell(hidden_size)
        elif rnn_cell_name == "lstm":
            return tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=state_is_tuple)
        elif rnn_cell_name == "mygru":
            return MyGRUCell(hidden_size)
        elif rnn_cell_name == "mylstm":
            return MyLSTMCell(hidden_size, state_is_tuple=state_is_tuple)


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        if self.config.my_use_word_embedding:
            with tf.variable_scope("words"):
                if self.config.embeddings is None:
                    self.logger.info("WARNING: randomly initializing word vectors")
                    _word_embeddings = tf.get_variable(
                            name="_word_embeddings",
                            dtype=tf.float32,
                            shape=[self.config.nwords, self.config.dim_word])
                else:
                    _word_embeddings = tf.Variable(
                            self.config.embeddings,
                            name="_word_embeddings",
                            dtype=tf.float32,
                            trainable=self.config.train_embeddings)

                word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                        self.word_ids, name="word_embeddings")

                if self.config.binary_weights_word:
                    self.add_binary_weights_op(_word_embeddings)

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                if not self.config.my_only_use_forward_char:
                    # bi lstm on chars
                    cell_fw = self.get_rnn_cell(self.config.hidden_size_char,
                                                state_is_tuple=True)
                    cell_bw = self.get_rnn_cell(self.config.hidden_size_char,
                                                state_is_tuple=True)
                    _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                    # read and concat output
                    if 'lstm' in self.config.my_rnn_cell.lower():
                        _, ((_, output_fw), (_, output_bw)) = _output
                    else:
                        _, (output_fw, output_bw) = _output
                    output = tf.concat([output_fw, output_bw], axis=-1)

                    # shape = (batch size, max sentence length, char hidden size * 2)
                    output = tf.reshape(output,
                                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                else: # only forward
                    cell = self.get_rnn_cell(self.config.hidden_size_char,
                                                      state_is_tuple=True)
                    _output = tf.nn.dynamic_rnn(
                        cell, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                    if "lstm" in self.config.my_rnn_cell.lower():
                        # read output, output is output of LSTM cell in last time step
                        _, (_, output_cell) = _output
                    else:
                        _, output_cell = _output

                    # shape = (batch size, max sentence length, char hidden size)
                    output = tf.reshape(output_cell, shape=[s[0], s[1], self.config.hidden_size_char])

                if self.config.my_use_word_embedding:
                    word_embeddings = tf.concat([word_embeddings, output], axis=-1)
                else:
                    word_embeddings = output

                if self.config.binary_weights_char:
                    self.add_binary_weights_op(_char_embeddings)

        with tf.variable_scope("dictionary"):
            if self.config.use_dict:
                dict_variable = tf.cast(self.dict_ids, tf.float32, name="dict_variable")
                word_embeddings = tf.concat([word_embeddings, dict_variable], axis=-1)

        with tf.variable_scope("letter_trigrams"):
            if self.config.use_letter_trigram:
                # get letter trigram embeddings matrix
                _ltg_embeddings = _letter_trigram_embeddings = tf.get_variable(
                    name="_letter_trigram_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nletter_trigrams, self.config.dim_letter_trigram])
                if self.config.letter_trigram_dummy_row_enabled:
                    dummy_row = tf.zeros([1, self.config.dim_letter_trigram])
                    _letter_trigram_embeddings =tf.concat([_ltg_embeddings, dummy_row], axis=0)
                letter_trigram_embeddings = tf.nn.embedding_lookup(_letter_trigram_embeddings,
                                               self.letter_trigram_ids, name="letter_trigram_embeddings")
                letter_trigram_embeddings_sum = tf.reduce_sum(letter_trigram_embeddings, axis=-2)
                word_embeddings = tf.concat([word_embeddings, letter_trigram_embeddings_sum], axis=-1)

                if self.config.binary_weights_ltg:
                    self.add_binary_weights_op(_ltg_embeddings)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_binary_weights_op(self, weights):
        _abs_weights = tf.abs(weights)
        _binary_value_weights = tf.sign(tf.sign(weights) + 0.5)
        _binary_weights = _binary_value_weights * tf.reduce_mean(_abs_weights)
        self.binary_weights_ops.append(tf.assign(weights, _binary_weights))

    def add_rnn_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        if self.config.embedding_proj:
            with tf.variable_scope("embedding_proj"):
                embedding_shape = self.word_embeddings.get_shape()
                embedding_proj_w = tf.get_variable("weight", dtype=tf.float32,
                                                   shape=[embedding_shape[-1], self.config.dim_word_proj])
                embedding_proj_b = tf.get_variable("bias", dtype=tf.float32, shape=[self.config.dim_word_proj])

                embedding = tf.transpose(self.word_embeddings, (1, 0, 2))
                embedding_shape = embedding.get_shape()
                time_steps = tf.shape(embedding)[0]
                proj_input_ta = tf.TensorArray(tf.float32, time_steps, tensor_array_name="embedding_proj_input_ta")
                proj_output_ta = tf.TensorArray(tf.float32, time_steps, tensor_array_name="embedding_proj_output_ta")
                proj_input_ta = proj_input_ta.unstack(embedding)

                def _time_step(time_, output_ta):
                    input_t = proj_input_ta.read(time_)
                    input_t.set_shape(embedding_shape.with_rank_at_least(3)[1:])
                    proj_embedding = tf.add(tf.matmul(input_t, embedding_proj_w), embedding_proj_b)
                    # add activation for projection layer
                    #proj_embedding = tf.nn.tanh(proj_embedding)

                    output_ta = output_ta.write(time_, proj_embedding)

                    return time_ + 1, output_ta

                time_0 = tf.constant(0, dtype=tf.int32, name="time")
                _, proj_output_ta = tf.while_loop(
                    cond=lambda _time, _: _time < time_steps,
                    body=_time_step,
                    loop_vars=(time_0, proj_output_ta),
                    parallel_iterations=32,
                    swap_memory=False
                )

                proj_outputs = proj_output_ta.stack()
                _word_embeddings = tf.transpose(proj_outputs, (1, 0, 2))
                self.word_embeddings_proj = tf.nn.dropout(_word_embeddings, self.dropout)
                self.word_embeddings = self.word_embeddings_proj

        with tf.variable_scope("bi-lstm"):
            if not self.config.my_only_use_forward_word:
                cell_fw = self.get_rnn_cell(self.config.hidden_size_lstm)
                cell_bw = self.get_rnn_cell(self.config.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.word_embeddings,
                        sequence_length=self.sequence_lengths, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
            else: # forward only
                cell = self.get_rnn_cell(self.config.hidden_size_lstm)
                output, _ = tf.nn.dynamic_rnn(
                    cell, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.nn.dropout(output, self.dropout)
            self.rnn_output = output

    def get_task_suffix(self, i=""):
        return "_task_{}".format(self.idx_to_task[i])

    def add_proj_op(self):
        input_dim = self.config.hidden_size_lstm
        if not self.config.my_only_use_forward_word:
            input_dim = 2 * input_dim

        with tf.variable_scope("states_agg"):
            if self.config.classifier_rnn_agg == 'last':
                nbatches = tf.shape(self.rnn_output)[0]
                indices = tf.stack([tf.range(nbatches), self.sequence_lengths - 1], axis=1)
                rnn_last_word = tf.gather_nd(self.rnn_output, indices)
                rnn_states_agg = rnn_last_word[:, :self.config.hidden_size_lstm]

                if not self.config.my_only_use_forward_word:
                    rnn_first_word = self.rnn_output[:, 0, :]
                    bw_rnn_first_word = rnn_first_word[:, self.config.hidden_size_lstm:]
                    rnn_states_agg = tf.concat([rnn_states_agg, bw_rnn_first_word], axis=-1)
            elif self.config.classifier_rnn_agg == 'avg':
                max_seq_len = tf.reduce_max(self.sequence_lengths)
                mask = tf.sequence_mask(self.sequence_lengths, maxlen=max_seq_len, dtype=tf.float32)
                mask = tf.expand_dims(mask, -1)
                rnn_states_agg = tf.reduce_sum(self.rnn_output * mask, axis=1) / tf.reduce_sum(mask, axis=1)
            elif self.config.classifier_rnn_agg == 'attention':
                rnn_states_agg = attention(self.rnn_output, self.config.attention_size_rnn_agg)
                # TODO: add dropout
            else:
                raise Exception('Unknown rnn aggregation method for classifier: {}'.
                                format(self.config.classifier_rnn_agg))

        with tf.variable_scope("proj_domain"):
            W = tf.get_variable("weight", dtype=tf.float32,
                                shape=[input_dim, self.config.ntasks])

            b = tf.get_variable("bias", shape=[self.config.ntasks],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            pred = tf.matmul(rnn_states_agg, W) + b
            self.domain_logits = pred

        self.tasks_intents_logits, self.tasks_labels_logits = [], []
        for i in range(self.config.ntasks):
            with tf.variable_scope("proj" + self.get_task_suffix(i)):
                with tf.variable_scope("intent"):
                    W = tf.get_variable("weight", dtype=tf.float32,
                                        shape=[input_dim, self.config.tasks_nintents[i]])

                    b = tf.get_variable("bias", shape=[self.config.tasks_nintents[i]],
                                        dtype=tf.float32, initializer=tf.zeros_initializer())

                    pred = tf.matmul(rnn_states_agg, W) + b
                    self.tasks_intents_logits.append(pred)

                with tf.variable_scope("label"):
                    W = tf.get_variable("weight", dtype=tf.float32,
                                        shape=[input_dim, self.config.tasks_ntags[i]])

                    b = tf.get_variable("bias", shape=[self.config.tasks_ntags[i]],
                            dtype=tf.float32, initializer=tf.zeros_initializer())

                    nsteps = tf.shape(self.rnn_output)[1]
                    output = tf.reshape(self.rnn_output, [-1, input_dim])
                    pred = tf.matmul(output, W) + b
                    self.tasks_labels_logits.append(tf.reshape(pred, [-1, nsteps, self.config.tasks_ntags[i]]))

                    # TODO: split this binary weights op for each task, and domain/intents
                    if self.config.binary_weights_proj:
                        # add binary weights projection layer, weight and bias
                        self.add_binary_weights_op(W)
                        self.add_binary_weights_op(b)


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        with tf.variable_scope("preds"):
            self.domains_pred = tf.cast(tf.argmax(self.domain_logits, axis=-1),
                                        tf.int32)

            self.tasks_intents_pred = []
            for i in range(self.config.ntasks):
                self.tasks_intents_pred.append(tf.cast(tf.argmax(self.tasks_intents_logits[i], axis=-1),
                                                       tf.int32))
            if not self.config.use_crf:
                self.tasks_labels_pred = []
                for i in range(self.config.ntasks):
                    self.tasks_labels_pred.append(tf.cast(tf.argmax(self.tasks_labels_logits[i], axis=-1),
                                                          tf.int32))

    def add_loss_op(self):
        """Defines the loss"""
        with tf.variable_scope("loss_domain"):
            if self.config.ntasks > 1:
                domain_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.domain_logits, labels=self.domains)
                domain_loss = tf.reduce_mean(domain_losses)
            else:
                domain_loss = tf.constant(0.0)

        self.tasks_loss = []
        self.tasks_summary = []
        if self.config.use_crf:
            self.tasks_trans_params = []

        for i in range(self.config.ntasks):
            with tf.variable_scope("loss" + self.get_task_suffix(i)):
                task_loss = domain_loss

                with tf.variable_scope("intent"):
                    if self.config.tasks_nintents[i] > 1:
                        intent_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.tasks_intents_logits[i], labels=self.intents)
                        weights = tf.reduce_sum(self.config.tasks_intents_weights[i] *
                                                tf.one_hot(self.intents, self.config.tasks_nintents[i]), axis=1)
                        task_loss += (tf.reduce_sum(weights * intent_losses) / tf.reduce_sum(weights))

                with tf.variable_scope("label"):
                    if self.config.use_crf:
                        trans_params = tf.get_variable("transitions",
                                                       shape=[self.config.tasks_ntags[i], self.config.tasks_ntags[i]])
                        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                                self.tasks_labels_logits[i], self.labels, self.sequence_lengths, trans_params)
                        self.tasks_trans_params.append(trans_params)  # need to evaluate it for decoding
                        tag_loss = tf.reduce_mean(-log_likelihood)
                    else:
                        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=self.tasks_labels_logits[i], labels=self.labels)
                        mask = tf.sequence_mask(self.sequence_lengths)
                        losses = tf.boolean_mask(losses, mask)
                        tag_loss = tf.reduce_mean(losses)

                    if self.config.tasks_ntags[i] > 1:
                        task_loss += tag_loss

                with tf.variable_scope("l2norm"):
                    if self.config.l2_lambda > 0:
                        for v in tf.global_variables():
                            if self.get_task_suffix() not in v.name or self.get_task_suffix(i) in v.name:
                                # add l2 loss to weights
                                if "weight" in v.name:
                                    task_loss += self.config.l2_lambda * tf.nn.l2_loss(v)
                                elif self.config.l2_bias and "bias" in v.name:
                                    task_loss += self.config.l2_lambda * tf.nn.l2_loss(v)
                                elif self.config.l2_l3g and "_letter_trigram_embeddings" in v.name:
                                    task_loss += self.config.l2_lambda * tf.nn.l2_loss(v)

                self.tasks_loss.append(task_loss)

                # for tensorboard
                self.tasks_summary.append(tf.summary.scalar("loss", task_loss))


    def build(self):
        if self.config.graph_random_seed is not None:
            tf.set_random_seed(self.config.graph_random_seed)
            self.logger.info('Graph random seed: {}'.format(self.config.graph_random_seed))

        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_rnn_op()
        self.add_proj_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.tasks_loss,
                          self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words, task_id):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            domains, intents, logits, trans_params = self.sess.run(
                    [self.domains_pred, self.tasks_intents_pred[task_id],
                     self.tasks_labels_logits[task_id], self.tasks_trans_params[task_id]], feed_dict=fd)

            # iterate over the sentences because no batching in viterbi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return domains, intents, viterbi_sequences, sequence_lengths

        else:
            domains, intents, labels_pred = self.sess.run(self.domains_pred, self.tasks_intents_pred,
                                                          self.tasks_labels_pred[task_id], feed_dict=fd)

            return domains, intents, labels_pred, sequence_lengths


    def run_epoch(self, trains, devs, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        batch_size = self.config.batch_size

        # merge trains
        random_seed = self.config.batch_sequence_random_seed
        if random_seed is not None:
            random_seed = random_seed * epoch
        train_minibatches = merge_datasets(trains, batch_size, random_seed, self.config.task_mb_merge)

        # progbar stuff for logging
        nbatches = len(train_minibatches)
        prog = Progbar(target=nbatches)

        # iterate over dataset
        prog_info = ''

        # set binary weights embeddings for binary training
        self.sess.run(self.binary_weights_ops)
        tasks_last_summary = [-(2**10)] * self.config.ntasks
        for i, (task_id, (task_mb_id, (intents, words, labels))) in enumerate(train_minibatches):
            fd, _ = self.get_feed_dict(words, [task_id] * len(intents), intents, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.tasks_train_op[task_id], self.tasks_loss[task_id], self.tasks_summary[task_id]], feed_dict=fd)
            prog_info = prog.update(i + 1, [("train loss" + self.get_task_suffix(task_id), train_loss)])

            # set binary weights embeddings for binary training
            self.sess.run(self.binary_weights_ops)

            # tensorboard
            if i - tasks_last_summary[task_id] >= 10:
                self.file_writer.add_summary(summary, epoch*nbatches + i)
                tasks_last_summary[task_id] = i

        self.logger.info(prog_info)
        self.logger.info('Mini batch sequence: ' +
                     ','.join(['{}_{}'.format(task_id, task_mb_id) for task_id, (task_mb_id, _) in train_minibatches]))

        f1_avg = 0
        for i in range(len(devs)):
            metrics = self.run_evaluate(devs[i], i)
            msg = self.get_metrics_msg(metrics)
            self.logger.info(msg)
            f1_avg += metrics["f1"]
        f1_avg /= len(devs)

        return f1_avg


    def run_evaluate(self, test, task_id, return_pred=False):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        metric_calc = MetricCalc(self.config.vocab_tasks, self.config.vocab_tasks_intents[task_id],
                                 self.config.vocab_tasks_tags[task_id], self.config.vocab_tasks_chunk_types[task_id])
        eval_start, eval_end, sent_num, total_padded_words_count = time.time(), time.time(), 0, 0
        preds = defaultdict(list)
        for intents, words, labels in minibatches(test, self.config.batch_size):
            domains_pred, intents_pred, labels_pred, sequence_lengths = self.predict_batch(words, task_id)
            eval_end = time.time()

            sent_num += len(sequence_lengths)
            total_padded_words_count += (len(sequence_lengths) * max(sequence_lengths))

            metric_calc.add_batch([task_id] * len(domains_pred), domains_pred,
                                  intents, intents_pred,
                                  labels, labels_pred, sequence_lengths)
            if return_pred:
                preds['pred_domains'].extend(domains_pred)
                preds['pred_intents'].extend(intents_pred)
                preds['pred_labels'].extend(
                    [lab_pred[:length] for lab_pred, length in zip(labels_pred, sequence_lengths)])

        eval_elapsed = 1000 * (eval_end - eval_start)

        metrics_dict = metric_calc.get_metrics()
        perf_dict = {"_eval_time_ms_elapsed": eval_elapsed,
                     "_eval_time_ms_per_sent": eval_elapsed / sent_num,
                     "_eval_time_ms_per_token": eval_elapsed / total_padded_words_count,
                     "_sent_num": sent_num,
                     "_total_padded_words_count": total_padded_words_count}
        result = {**metrics_dict, **perf_dict}
        if return_pred:
            result = {**result, **preds}
        return result

    def get_metrics_msg(self, metrics):
        msg_num = []
        msg_table = []
        for k, v in metrics.items():
            if (isinstance(v, (int, float))):
                msg_num.append("{} {:04.2f}".format(k, v))
            elif (isinstance(v, tuple) and len(v) == 3 and isinstance(v[2], np.ndarray)):
                str_array = np.array(["{:04.2f}".format(x) for x in v[2].reshape(v[2].size)])
                str_array = str_array.reshape(v[2].shape)
                str_array = np.vstack([np.array(v[1]).reshape(1, -1), str_array])
                str_array = np.hstack([np.array([''] + v[0]).reshape(-1, 1), str_array])
                msg_table.append('-----' + k + '-----\n' + tabulate(str_array, headers='firstrow'))

        return '\n'.join(sorted(msg_num) + sorted(msg_table))

    def predict(self, words_raw, task_id):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        dataset = get_CoNLL_dataset(create_memory_file_from_words(words_raw), self.config, task_id)
        _, x_batch, _ = next(minibatches(dataset, 1))
        pred_domains, pred_intents, pred_ids, _ = self.predict_batch(x_batch, task_id)

        pred_domain_txt = self.idx_to_task[pred_domains[0]]
        pred_intent_txt = self.tasks_idx_to_intent[task_id][pred_intents[0]]
        pred_tags_txt = [self.tasks_idx_to_tag[task_id][idx] for idx in list(pred_ids[0])]

        return pred_domain_txt, pred_intent_txt, pred_tags_txt

    def get_embedding_projection(self, words):
        feed, sequence_length = self.get_feed_dict(words, dropout=1.0)
        embedding_projection = self.sess.run(self.word_embeddings_proj, feed_dict=feed)
        return embedding_projection

    def export_ndarray(self, f, name, array):
        array = np.squeeze(array, axis=0)
        f.write('{} {} {}\n'.format(name, array.shape[0], array.shape[1]))
        for row in array:
            f.write('{}\n'.format(' '.join([str(num) for num in row])))

    def export_layer_result(self, words_raw, task_id):
        dataset = get_CoNLL_dataset(create_memory_file_from_words(words_raw), self.config, task_id)
        x_batch, _ = next(minibatches(dataset, 1))

        fd, _ = self.get_feed_dict(x_batch, dropout=1.0)
        word_embedding, rnn_output, logits = \
            self.sess.run([self.word_embeddings, self.rnn_output, self.tasks_labels_logits[task_id]], feed_dict=fd)

        with open(self.config.path_layer_result, 'a') as f:
            word_normalize = get_processing_word()
            f.write('\n{}\n'.format(' '.join([word_normalize(w) for w in words_raw])))
            self.export_ndarray(f, 'word_embedding', word_embedding)
            self.export_ndarray(f, 'rnn_output', rnn_output)
            self.export_ndarray(f, 'logits', logits)
            f.write('\n')