import argparse
import glob
import os

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, get_dict_trie, get_processing_dict, get_name_for_task, \
    get_ordered_keys, get_class_weights


class Config(object):
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        self.set_app_arguments()

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        for att in self.__dir__():
            if not att.endswith("__"):
                self.logger.info("{}:{}".format(att, self.__getattribute__(att)))


        # load if requested (default)
        if load:
            self.load()

    def set_app_arguments(self):
        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        parser = argparse.ArgumentParser()
        parser.add_argument("--dim-char", type=int)
        parser.add_argument("--dim-letter-trigram", type=int)
        parser.add_argument("--dim-word-proj", type=int)
        parser.add_argument("--use-pretrained", type=str2bool)
        parser.add_argument("--trie-separator")
        parser.add_argument("--train-embeddings", type=str2bool)
        parser.add_argument("--max-iter", type=int)
        parser.add_argument("--nepochs", type=int)
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--batch-sequence-random-seed", type=int)
        parser.add_argument("--lr-method")
        parser.add_argument("--lr", type=float)
        parser.add_argument("--lr-decay", type=float)
        parser.add_argument("--clip", type=int)
        parser.add_argument("--nepoch-no-imprv", type=int)
        parser.add_argument("--l2-lambda", type=int)
        parser.add_argument("--l2-bias", type=str2bool)
        parser.add_argument("--l2-l3g", type=str2bool)
        parser.add_argument("--hidden-size-char", type=int)
        parser.add_argument("--hidden-size-lstm", type=int)
        parser.add_argument("--use-crf", type=str2bool)
        parser.add_argument("--use-chars", type=str2bool)
        parser.add_argument("--use-letter-trigram", type=str2bool)
        parser.add_argument("--letter-trigram-dummy-row-enabled", type=str2bool)
        parser.add_argument("--use-dict", type=str2bool)
        parser.add_argument("--chars-lowercase", type=str2bool)
        parser.add_argument("--trimmed-word-num", type=int)
        parser.add_argument("--my-only-use-forward-char", type=str2bool)
        parser.add_argument("--my-only-use-forward-word", type=str2bool)
        parser.add_argument("--my-use-word-embedding", type=str2bool)
        parser.add_argument("--my-rnn-cell")
        parser.add_argument("--graph-random-seed", type=int)
        parser.add_argument("--tfdbg-enabled", type=str2bool)
        parser.add_argument("--embedding-proj", type=str2bool)
        parser.add_argument("--max-sent-len", type=int)
        parser.add_argument("--max-word-len", type=int)
        parser.add_argument("--binary-weights-word", type=str2bool)
        parser.add_argument("--binary-weights-char", type=str2bool)
        parser.add_argument("--binary-weights-ltg", type=str2bool)
        parser.add_argument("--binary-weights-proj", type=str2bool)
        (args, unknown) = parser.parse_known_args()
        for att in self.__dir__():
            if att in args.__dir__() and not att.endswith("__"):
                if args.__getattribute__(att) is not None:
                    self.__setattr__(att, args.__getattribute__(att))

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_tasks = load_vocab(self.filename_task_names)
        self.task_names = get_ordered_keys(self.vocab_tasks)

        self.vocab_words = load_vocab(self.filename_words)
        self.filename_tasks_tags = [get_name_for_task(self.filename_tasks_tags, name) for name in self.task_names]
        self.vocab_tasks_tags = [load_vocab(filename_tags) for filename_tags in self.filename_tasks_tags]
        self.vocab_letter_trigrams = load_vocab(self.filename_letter_trigrams)
        self.filename_tasks_chunk_types = \
            [get_name_for_task(self.filename_tasks_chunk_types, name) for name in self.task_names]
        self.vocab_tasks_chunk_types = [load_vocab(filename) for filename in self.filename_tasks_chunk_types]
        self.filename_tasks_intents = [get_name_for_task(self.filename_tasks_intents, name) for name in self.task_names]
        self.vocab_tasks_intents = [load_vocab(filename) for filename in self.filename_tasks_intents]
        self.filename_tasks_intents_weights = \
            [get_name_for_task(self.filename_tasks_intents_weights, name) for name in self.task_names]
        self.tasks_intents_weights = [get_class_weights(filename, len(vocab)) for filename, vocab in
                                      zip(self.filename_tasks_intents_weights, self.vocab_tasks_intents)]
        self.vocab_chars = load_vocab(self.filename_chars)
        self.vocab_dict_types = load_vocab(self.filename_dict_types)

        self.ntasks = len(self.vocab_tasks)
        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.tasks_ntags = [len(vocab_tags) for vocab_tags in self.vocab_tasks_tags]
        self.nletter_trigrams = len(self.vocab_letter_trigrams)
        self.tasks_nchunk_types = [len(vocab_chunk_types) for vocab_chunk_types in self.vocab_tasks_chunk_types]
        self.tasks_nintents = [len(vocab) for vocab in self.vocab_tasks_intents]
        self.ndict_types = len(self.vocab_dict_types)

        # 2. get processing functions that map str -> id
        # we always extract char ids and letter trigram ids,
        # and decide whether to add these features in NN building,
        # so we can add new features easily
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars, self.vocab_letter_trigrams, lowercase=True,
                                                   chars=True, chars_lowercase=self.chars_lowercase,
                                                   letter_trigrams=True, max_word_len=self.max_word_len)
        self.processing_tasks_tag = [get_processing_word(vocab_tags,
                                        lowercase=False, allow_unk=False) for vocab_tags in self.vocab_tasks_tags]
        self.processing_task_intents = [get_processing_word(vocab,
                                        lowercase=False, allow_unk=False) for vocab in self.vocab_tasks_intents]
        self.dict_trie, _, _ = get_dict_trie(self.filename_dict,
                                          get_processing_word(self.vocab_words, lowercase=True),
                                          get_processing_word(self.vocab_dict_types, allow_unk=False),
                                          self.trie_separator)
        self.processing_dict = get_processing_dict(self.dict_trie,
                                                   self.ndict_types,
                                                   self.trie_separator)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                           if self.use_pretrained else None)

        # 4. get embedding projection
        self.projection_embedding = (get_trimmed_glove_vectors(self.filename_embedding_projection_npz)
                                     if self.use_embedding_proj_pred else None)

    # Arguments for Philly run which will overwrite config
    # These config value will be used as part of other configs so we should set them before.
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-model-path")
    parser.add_argument("--input-training-data-path")
    parser.add_argument("--input-previous-model-path")
    parser.add_argument("--input-validation-data-path")
    parser.add_argument("--word-embedding-name")
    parser.add_argument("--dim-word", type=int)
    (args, unknown) = parser.parse_known_args()

    # Inputs of build_data.py, train.py and evaluate.py
    dir_data = "data/sentiment/" if not args.input_training_data_path else args.input_training_data_path
    filename_trains = filename_devs = sorted(glob.glob(os.path.join(dir_data, "train*.txt")))
    filename_test_ins = sorted(glob.glob(os.path.join(dir_data, "test/test_in*.txt")))
    filename_test_outs = sorted(glob.glob(os.path.join(dir_data, "test/test_out*.txt")))
    filename_dict = dir_data + "dict.txt"
    if args.input_training_data_path:
        filename_1p2gb_sample = args.input_training_data_path + "test_1.2GB.sample.0.03.DummyLabeled.txt"
    else:
        filename_1p2gb_sample = "data/test_1.2GB.sample.0.03.DummyLabeled.txt"

    # Inputs of build_data.py
    dir_word_embedding = "data/" if not args.input_training_data_path else args.input_training_data_path
    dim_word = 64 if not args.dim_word else args.dim_word
    word_embedding_name = "baike.1M" if not args.word_embedding_name else args.word_embedding_name
    filename_glove = "{}/{}/{}.{}d.txt".format(dir_word_embedding, word_embedding_name, word_embedding_name, dim_word)

    # Outputs of build_data.py and inputs of train.py and evaluate.py
    dir_build_data_output = dir_data
    if args.input_validation_data_path:
        dir_build_data_output = args.input_validation_data_path + '/'  # When Philly run evaluate.py
    elif args.input_previous_model_path:
        dir_build_data_output = args.input_previous_model_path + '/'  # When Philly run train.py
    elif args.output_model_path:
        dir_build_data_output = args.output_model_path + '/'  # When Philly run build_data.py
    # vocab (created from dataset with build_data.py)
    filename_words = dir_build_data_output + "words.txt"
    filename_letter_trigrams = dir_build_data_output + "letter_trigrams.txt"
    filename_task_names = dir_build_data_output + "tasks.txt"
    filename_tasks_intents = os.path.join(dir_build_data_output, "intents")
    filename_tasks_intents_weights = os.path.join(dir_build_data_output, "intents.weights")
    filename_tasks_tags = os.path.join(dir_build_data_output, "tags")
    filename_tasks_chunk_types = os.path.join(dir_build_data_output, "chunk_types")
    filename_chars = dir_build_data_output + "chars.txt"
    filename_dict_types = dir_build_data_output + "dict_types.txt"
    filename_dict_paths = dir_build_data_output + "dict_paths.txt"

    # trimmed embeddings (created from filename_glove with build_data.py)
    filename_trimmed = dir_build_data_output + "{}.{}d.trimmed.npz".format(word_embedding_name, dim_word)

    # Output of train.py
    dir_output = "results/test/" if not args.output_model_path else args.output_model_path + '/'
    dir_model = dir_output + "model.weights/"
    path_log = dir_output + "log.txt"

    # Inputs of evaluate.py and outputs of train.py
    dir_model_evaluate = dir_model if not args.input_previous_model_path else args.input_previous_model_path + "/model.weights/"

    # training
    dim_char = 100
    dim_letter_trigram = 50
    dim_word_proj = 50
    trie_separator = '.'
    max_iter = None  # if not None, max number of examples in Dataset
    use_pretrained = True
    train_embeddings = False
    nepochs = 30
    dropout = 0.4
    batch_size = 20
    batch_sequence_random_seed = None
    task_mb_merge = 'permute'  # "permute", "cycle"
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 3
    l2_lambda = 0  # set l2_lambda to 0 to not use l2 loss
    l2_bias = False
    l2_l3g = False

    # model hyperparameters
    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 300  # lstm on word embeddings
    attention_size_rnn_agg = int(hidden_size_lstm / 3)

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU
    use_letter_trigram = False  # whether to use letter trigram features
    letter_trigram_dummy_row_enabled = True  # whether to add dummy row for l3t embedding
    use_dict = False  # whether to use dictionary feature
    classifier_rnn_agg = "last"  # "last", "avg", "attention"

    chars_lowercase = True  # if char_lowercase, only use lowercase char
    trimmed_word_num = 0  # if set 0, no trim by default, or trim words to min(trimmed_word_num, Config.nwords)

    my_only_use_forward_char = False  # only use forward rnn in char
    my_only_use_forward_word = False  # only use forward rnn in word
    my_use_word_embedding = True  # use word embedding, if False, use_chars must be True
    my_rnn_cell = "MyGRU"  # available candidate: "GRU", "LSTM", "MyGRU", "MyLSTM";
    # "My*" is cells that make internal variable clearly

    # if not None, will be set in tf.set_random_seed().
    # with this, the result will be same for same graph
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    # https://stackoverflow.com/questions/36096386/tensorflow-set-random-seed-not-working
    graph_random_seed = None
    # tfdbg
    tfdbg_enabled = False

    embedding_proj = False


    max_sent_len = None  # 99% 107, 99.9% 256, 99.99% 413, 99.999% 508
    max_word_len = None  # 99% 11, 99.9% 15, 99.99% 27, 99.999% 81

    binary_weights_word = False
    binary_weights_char = False
    binary_weights_ltg = False
    binary_weights_proj = False

    # Retired config
    filename_embedding_projection = "data/embedding_projection.{}d.txt".format(dim_word_proj)
    filename_embedding_projection_npz = "data/embedding_projection.{}d.trimmed.npz".format(dim_word_proj)
    dir_proj_model = dir_output + "proj_model.weights/"
    use_embedding_proj_pred = False
