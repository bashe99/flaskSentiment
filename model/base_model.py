import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None


    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_train_op(self, lr_method, lr, tasks_loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower() # lower to make sure

        self.tasks_train_op = []
        with tf.variable_scope("train_step"):
            for i in range(self.config.ntasks):
                if _lr_m == 'adam': # sgd method
                    optimizer = tf.train.AdamOptimizer(lr)
                elif _lr_m == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(lr)
                elif _lr_m == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(lr)
                elif _lr_m == 'rmsprop':
                    optimizer = tf.train.RMSPropOptimizer(lr)
                else:
                    raise NotImplementedError("Unknown method {}".format(_lr_m))

                if clip > 0: # gradient clipping if clip is positive
                    grads, vs     = zip(*optimizer.compute_gradients(tasks_loss[i]))
                    grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                    self.tasks_train_op.append(optimizer.apply_gradients(zip(grads, vs)))
                else:
                    self.tasks_train_op.append(optimizer.minimize(tasks_loss[i]))


    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)
        if self.config.tfdbg_enabled:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(tf.global_variables_initializer())
        # if max_to_keep = None, although it saves all models,
        # but all_model_checkpoint_paths contains just the last one,
        # check tensorflow _RecordLastCheckpoint code
        self.saver = tf.train.Saver(max_to_keep=100)


    def restore_latest_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        self.restore_session(tf.train.latest_checkpoint(dir_model))


    def restore_session(self, model_path):
        self.logger.info("Loading model {}...".format(model_path))
        self.saver.restore(self.sess, model_path)


    def get_all_checkpoints(self, dir_model):
        self.logger.info("Finding all model checkpoints...")
        state = tf.train.get_checkpoint_state(dir_model)
        return state.all_model_checkpoint_paths


    def save_session(self, epoch):
        """Saves session = weights"""
        self.saver.save(self.sess, self.config.dir_model, global_step=epoch)


    def prepare_model_dir(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        else:
            for file_name in os.listdir(self.config.dir_model):
                os.remove(os.path.join(self.config.dir_model, file_name))


    def close_session(self):
        """Closes the session"""
        self.sess.close()


    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                                                 self.sess.graph)


    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        self.prepare_model_dir()

        best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        self.add_summary() # tensorboard

        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                        self.config.nepochs))

            score = self.run_epoch(train, dev, epoch)
            self.config.lr *= self.config.lr_decay # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session(epoch)
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break


    def evaluate(self, test, task_id, return_pred=False):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set {}".format(test.file))
        metrics = self.run_evaluate(test, task_id, return_pred)
        msg = self.get_metrics_msg(metrics)
        self.logger.info(msg)

        return metrics
