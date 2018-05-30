import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from .ner_model import NERModel


class ProjectionNERModel(NERModel):
    """Specialized class of Model for NER"""

    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            _projection_embeddings = tf.Variable(
                self.config.projection_embedding,
                name="_proj_embeddings",
                dtype=tf.float32,
                trainable=False)
            self.word_embeddings = tf.nn.embedding_lookup(_projection_embeddings, self.word_ids,
                                                          name="projection_embeddings")

    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
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

        with tf.variable_scope("proj"):
            if not self.config.my_only_use_forward_word:
                W = tf.get_variable("weight", dtype=tf.float32,
                        shape=[2*self.config.hidden_size_lstm, self.config.ntags])
            else:
                W = tf.get_variable("weight", dtype=tf.float32,
                        shape=[self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("bias", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            if not self.config.my_only_use_forward_word:
                output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            else:
                output = tf.reshape(output, [-1, self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)
        if self.config.tfdbg_enabled:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(tf.global_variables_initializer())

        var_list = []
        for v in tf.trainable_variables():
            if "proj_embeddings" in v.name:
                continue
            var_list.append(v)

        self.saver = tf.train.Saver(var_list=var_list)

    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        saver = tf.train.Saver()
        saver.save(self.sess, self.config.dir_proj_model)

    def restore_session(self, model_path):
        self.logger.info("Loading model {}...".format(model_path))
        self.saver.restore(self.sess, model_path)
        # save projection embedding model
        self.logger.info("Saving projection model...")
        self.save_session()
