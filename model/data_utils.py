import io
import numpy as np
import os
import pygtrie
import tempfile


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, file, processing_word=None, processing_tag=None,
                 processing_dict=None, processing_intent=None, max_iter=None, max_sent_len=None):
        """
        Args:
            file: a path to text file or a file object
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            processing_dict: (optional) function to takes a sentence as input
            max_iter: (optional) max number of sentences to yield

        """
        self.file = file
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.processing_dict = processing_dict
        self.processing_intent = processing_intent
        self.max_iter = max_iter
        self.length = None
        self.max_sent_len = max_sent_len


    def __iter__(self):
        niter = 0
        with open(self.file, encoding='utf-8') if isinstance(self.file, str) else self.file as f:
            intent, words, tags = '', [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break

                        # add dictionary feature
                        if self.processing_dict is not None:
                            # for word processing, we expect all ids
                            # (letter trigram id, char id and word id, ...) are extracted
                            if len(words) > 0 and type(words[0]) is not tuple:
                                raise Exception("Unexpected, word is not a tuple")
                            word_ids = [word[-1] for word in words]
                            dict_ids = self.processing_dict(word_ids)
                            words = list(map(lambda w, d: ((d,) + w), words, dict_ids))

                        # max_sent_len
                        if self.max_sent_len is not None:
                            words = words[:self.max_sent_len]
                            tags = tags[:self.max_sent_len]

                        # intent
                        if not intent:
                            intent = 'none'
                        if self.processing_intent is not None:
                            intent = self.processing_intent(intent)

                        yield intent, words, tags
                        intent, words, tags = '', [], []
                else:
                    ls = line.split(' ')
                    if len(ls) == 1:
                        if len(intent) != 0:
                            raise Exception('Unexpected line: {}'.format(line))
                        else:
                            intent = line
                    else:
                        word, tag = ls[0],ls[-1]
                        if self.processing_word is not None:
                            word = self.processing_word(word)
                        if self.processing_tag is not None:
                            tag = self.processing_tag(tag)
                        words += [word]
                        tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_CoNLL_dataset(filename, config, task_id):
    return CoNLLDataset(filename, config.processing_word,
                        config.processing_tasks_tag[task_id],
                        config.processing_dict, config.processing_task_intents[task_id],
                        config.max_iter, config.max_sent_len)

def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_intents = set()
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for intent, words, tags in dataset:
            vocab_intents.add(intent)
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_intents, vocab_words, vocab_tags


def get_letter_trigrams(word):
    bounded_word = '#' + word + '#'
    letter_trigrams = [bounded_word[i:i+3] for i in range(len(bounded_word) - 2)]
    # to remove cases like " 16" in "+65 6272 1626" (not ascii space)
    letter_trigrams = [t for t in letter_trigrams if len(t.strip()) == 3]
    return letter_trigrams


def get_letter_trigram_vocab(vocab_words):
    vocab_letter_trigrams = set()
    for word in vocab_words:
        vocab_letter_trigrams.update(get_letter_trigrams(word))
    return vocab_letter_trigrams


def get_chunk_vocab(vocab_tags):
    vocab_chunk_types = set()
    for tag in vocab_tags:
        _, chunk_type = get_chunk_type_from_name(tag)
        vocab_chunk_types.add(chunk_type)
    return vocab_chunk_types


def get_char_vocab(datasets, chars_lowercase=False):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for dataset in datasets:
        for _, words, _ in dataset:
            for word in words:
                if chars_lowercase:
                    word = word.lower()
                vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            # print(line.split(' ')[0])
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def get_class_weights(filename, classes_num=None):
    if os.path.exists(filename):
        weights = []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    weights.append(float(line))
        return weights
    elif classes_num is not None:
        return [1.0] * classes_num
    else:
        raise Exception('Invalid class weights: {}'.format(filename))


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename, encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def trim_words(word_set, data_sets, num):
    """
    trim words number to num
    Args:
        word_set: word set
        data_sets: data set list
        num: trim number
    """
    word_dict = {}
    for data in data_sets:
        for word_list, _ in data:
            for word in word_list:
                if word not in word_set:
                    continue
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    sorted_list = sorted(word_dict.keys(), key=lambda w: word_dict[w], reverse=True)

    result_set = set()
    result_set.update(sorted_list[:num])
    return result_set


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def concate_list_and_tuple(a_list, b_num_or_tuple):
    if type(b_num_or_tuple) is tuple:
        result = a_list, *b_num_or_tuple
    else:
        result = a_list, b_num_or_tuple
    return result


def get_processing_word(vocab_words=None, vocab_chars=None,
                        vocab_letter_trigrams=None, lowercase=False,
                        chars=False, chars_lowercase=False, letter_trigrams=False, allow_unk=True, max_word_len=None):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars:
            char_ids = []
            char_word = word
            if chars_lowercase:
                char_word = char_word.lower()
            for char in char_word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]
            if max_word_len is not None:
                char_ids = char_ids[:max_word_len]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of letter trigrams
        if vocab_letter_trigrams is not None and letter_trigrams == True:
            letter_trigram_ids = []
            for l3t in get_letter_trigrams(word):
                # ignore letter trigrams out of vocabulary
                if l3t in vocab_letter_trigrams:
                    letter_trigram_ids += [vocab_letter_trigrams[l3t]]
            if max_word_len is not None:
                letter_trigram_ids = letter_trigram_ids[:max_word_len]

        # 3. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                #else:
                #    raise Exception("Unknown key is not allowed. Check that "\
                #                    "your vocab (tags?) is correct")

        # 4. return tuple: letter trigram ids, char ids, word id
        result = word
        if vocab_chars is not None and chars == True:
            result = concate_list_and_tuple(char_ids, result)

        if vocab_letter_trigrams is not None and letter_trigrams == True:
            result = concate_list_and_tuple(letter_trigram_ids, result)

        return result

    return f


def get_processing_dict(trie, ndict_types, trie_separator='.s'):
    def f(word_ids):
        word_ids = [str(word_id) for word_id in word_ids]
        dict_feat = [[0] * 2 * ndict_types for word_id in word_ids]
        for i in range(len(word_ids)):
            sent = trie_separator.join(word_ids[i:])
            prefix, dict_type = trie.longest_prefix(sent)
            if dict_type is not None:
                dict_feat[i][2 * dict_type] = 1
                for j in range(1, len(prefix.split(trie_separator))):
                    dict_feat[i + j][2 * dict_type + 1] = 1
        return tuple(dict_feat)

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    intent_batch, x_batch, y_batch = [], [], []
    for (intent, x, y) in data:
        if len(x_batch) == minibatch_size:
            yield intent_batch, x_batch, y_batch
            intent_batch, x_batch, y_batch = [], [], []

        if type(x[0]) == tuple:
            x = list(zip(*x))
        intent_batch.append(intent)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield intent_batch, x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    return get_chunk_type_from_name(tag_name)


def get_chunk_type_from_name(tag_name):
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def get_all_chunks(seq, tags):
    """Also include O chunk

        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3, "O": 0}
            result = [("PER", 0, 2), ('O', 2, 3) ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
        if tok_chunk_class == "B" or tok_chunk_type != chunk_type:
            if chunk_type is not None and chunk_start is not None:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)

            chunk_type = tok_chunk_type
            chunk_start = i

    if chunk_type is not None and chunk_start is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    # verify
    if set([c for c in chunks if c[0] != NONE]) != set(get_chunks(seq, tags)):
        raise Exception("Result of get_all_chunks is inconsistent with get_chunks")

    return chunks


def get_pr(metrics):
    out_metrics = np.vstack([
        np.divide(metrics[:, 0], metrics[:, 1], out=np.zeros_like(metrics[:, 0]), where=metrics[:, 0]!=0),
        np.divide(metrics[:, 0],  metrics[:, 2], out=np.zeros_like(metrics[:, 0]), where=metrics[:, 0]!=0)
    ]).transpose()
    divisor = 2 * np.multiply(out_metrics[:, 0], out_metrics[:, 1])
    dividend = np.add(out_metrics[:, 0], out_metrics[:, 1])
    out_metrics = np.hstack([
        out_metrics,
        np.divide(divisor, dividend, out=np.zeros_like(divisor), where=dividend!=0).reshape(-1, 1)
    ])

    return out_metrics


def get_ordered_keys(dictionary):
    return [e[0] for e in sorted(dictionary.items(), key=lambda e: e[1])]


def get_dict_trie(dict_file_name, processing_word=None, processing_dict_type=None, trie_separator='.'):
    trie = pygtrie.StringTrie(separator=trie_separator)
    paths = []
    dict_types = set()
    UNK_word_id = processing_word(UNK) if processing_word is not None else -1
    with open(dict_file_name, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sent, dict_type = line.split('\t')
                if processing_word is not None:
                    word_ids = [processing_word(word) for word in sent.split(' ')]
                    if UNK_word_id in word_ids:
                        continue
                    sent = trie_separator.join([str(word_id) for word_id in word_ids])
                if processing_dict_type is not None:
                    dict_type = processing_dict_type(dict_type)
                trie[sent] = dict_type
                paths.append('{}\t{}'.format(sent, dict_type))
                dict_types.add(dict_type)
    return trie, paths, list(dict_types)


def create_memory_file_from_words(words):
    return io.StringIO('{}\n\n'.format('\n'.join(['{} O'.format(w) for w in words])))


def get_task_vocab(filenames):
    if all('_' in filename for filename in filenames):
        return [filename.rsplit('.', 1)[0].rsplit('_', 1)[1] for filename in filenames]
    else:
        return list(str(i) for i in range(len(filenames)))


def get_name_for_task(prefix, task_name):
    return "{}_{}.txt".format(prefix, task_name)


def merge_lists_alternate(lists, len_per_list):
    result = []
    list_num = len(lists)
    indexes = [0] * list_num
    list_lens = [len(l) for l in lists]
    for _ in range(len_per_list):
        for i in range(list_num):
            result.append(lists[i][indexes[i]])
            indexes[i] = (indexes[i] + 1) % list_lens[i]

    return result


def merge_datasets(datasets, batch_size, random_seed, mode):
    datasets_mbs = []
    for i, dataset in enumerate(datasets):
        datasets_mbs.append([(i, mb_enum) for mb_enum in enumerate(minibatches(dataset, batch_size))])

    if mode == 'permute':
        first_mbs = [mbs[0] for mbs in datasets_mbs]
        remaining_mbs = [mb for mbs in datasets_mbs for mb in mbs[1:]]
        np.random.RandomState(seed=random_seed).shuffle(remaining_mbs)
        return first_mbs + remaining_mbs
    elif mode == 'cycle':
        for mbs in datasets_mbs:
            np.random.RandomState(seed=random_seed).shuffle(mbs)
        return merge_lists_alternate(datasets_mbs, max(len(mbs) for mbs in datasets_mbs))
    else:
        raise Exception('Unsupported mode: {}'.format(mode))


if __name__ == "__main__":
    seq = [4, 5, 0, 3]
    tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3, "O": 0}
    # Expected: [("PER", 0, 2), ('O', 2, 3) ("LOC", 3, 4)]
    result = get_all_chunks(seq, tags)
    print(result)
    print([c for c in result if c[0] != NONE] == get_chunks(seq, tags))

    seq = [4, 5, 5, 4, 0, 0, 3, 5, 3]
    tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3, "O": 0}
    # Expected: [('PER', 0, 3), ('PER', 3, 4), ('O', 4, 6), ('LOC', 6, 7), ('PER', 7, 8), ('LOC', 8, 9)]
    result = get_all_chunks(seq, tags)
    print(result)
    print([c for c in result if c[0] != NONE] == get_chunks(seq, tags))

    tmp_dict_filename = ''
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("The big bang theory\tTV\n")
        tmp.write("Game of the thrones\tTV\n")
        tmp.write("Angry Birds\tMOVIE\n")
        tmp_dict_filename = tmp.name
    trie, _, dict_types = get_dict_trie(tmp_dict_filename)
    print(trie)
    assert(set(dict_types) == set(['MOVIE', 'TV']))

    vocab_words = {'big': 0, 'bang': 1, 'the': 2, 'theory': 3, UNK: 4}
    processing_words = get_processing_word(vocab_words, lowercase=True, allow_unk=True)
    vocab_dict_types = {'MOVIE': 0, 'TV': 1}
    processing_dict_type = get_processing_word(vocab_dict_types)
    trie, _, dict_types = get_dict_trie(tmp_dict_filename, processing_words, processing_dict_type)
    print(trie)
    assert(set(dict_types) == set([0, 1]))

    words = [([0], 1), ([0, 1], 5), ([0, 1, 2], 3), ([0, 1, 2, 3], 5),
             ([0, 1, 2, 3, 4], 4), ([0, 1, 2, 3, 4, 5], 62),
             ([0, 1, 2, 3, 4, 5, 6], 9)]
    sep = '.'
    trie = trie = pygtrie.StringTrie(separator=sep)
    trie['3.5'] = 1
    trie['3.5.4'] = 1
    trie['3.5.4.6'] = 1
    processing_dict = get_processing_dict(trie, 2, sep)
    dict_ids = processing_dict([word[-1] for word in words])
    print(dict_ids)
    print(list(map(lambda w, d: ((d,) + w), words, dict_ids)))