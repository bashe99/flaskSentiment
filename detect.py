from model.data_utils import CoNLLDataset, get_CoNLL_dataset
from model.ner_model import NERModel
from model.embedding_projection_ner_model import ProjectionNERModel
from model.config import Config
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
import sys


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned

def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        data = {"input": words_raw}
        pred_intents = []
        for i in range(model.config.ntasks):
            pred_domain, pred_intent, pred_tags = model.predict(words_raw, i)
            # model.export_layer_result(words_raw, i)
            pred_intents.append(pred_intent)
            data['output' + model.get_task_suffix(i)] = pred_tags

        model.logger.info('domain: {}'.format(pred_domain))
        model.logger.info('intents: {}'.format(', '.join(pred_intents)))
        to_print = align_data(data)
        for key, seq in to_print.items():
            model.logger.info(seq)


def save_result_batchmode(model, filename, task_id):
    test = get_CoNLL_dataset(filename, model.config, task_id)
    if len(test) <= 0:
        return

    result = model.evaluate(test, task_id, return_pred=True)
    pred_domains = result['pred_domains']
    pred_intents = result['pred_intents']
    pred_labels = result['pred_labels']

    test = CoNLLDataset(filename, None, None, None, None, model.config.max_iter, model.config.max_sent_len)
    idx_to_domains = {idx: task for task, idx in model.config.vocab_tasks.items()}
    idx_to_intents = {idx: intent for intent, idx in model.config.vocab_tasks_intents[task_id].items()}
    vocab_tags = model.config.vocab_tasks_tags[task_id]
    idx_to_tag = {idx: tag for tag, idx in vocab_tags.items()}
    predict_result_filename = os.path.join(model.config.dir_output, 'predict.{}'.format(os.path.basename(filename)))
    with open(predict_result_filename, 'w', encoding='utf-8') as fout:
        num, triggered_num = 0, 0
        for _, words, tags in test:
            if model.config.ntasks > 1:
                fout.write('{}\n'.format(idx_to_domains[pred_domains[num]]))

            if model.config.tasks_nintents[task_id] > 1:
                fout.write('{}\n'.format(idx_to_intents[pred_intents[num]]))

            for word, pred in zip(words, pred_labels[num]):
                fout.write('{} {}\n'.format(word, idx_to_tag[pred]))
            fout.write('\n')

            triggered_num += any(l != vocab_tags['O'] for l in pred_labels[num])
            num += 1
        model.logger.info('Total sentences: {}, triggered: {}, ratio: {:.4f}'.
                          format(num, triggered_num, triggered_num / num))


def pickle_save(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

globalModel = None

def detectOnce(word):
    global globalModel
    pred_domain, pred_intent, pred_tags = globalModel.predict(word, 0)
    return pred_intent


def buildModel():
    global globalModel
    config = Config()
    # build model
    if config.use_embedding_proj_pred:
        globalModel = ProjectionNERModel(config)
    else:
        globalModel = NERModel(config)
    globalModel.build()
    globalModel.restore_latest_session(config.dir_model_evaluate)
    return True

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else None

    # create instance of config
    config = Config()

    # build model
    if config.use_embedding_proj_pred:
        globalModel = ProjectionNERModel(config)
    else:
        globalModel = NERModel(config)
    globalModel.build()

    # create dataset
    train = get_CoNLL_dataset(config.filename_trains[0], config, 0)
    test_in = get_CoNLL_dataset(config.filename_test_ins[0], config, 0)
    test_out = get_CoNLL_dataset(config.filename_test_outs[0], config, 0)

    if mode == 'inter':
        # interact
        model.restore_latest_session(config.dir_model_evaluate)
        interactive_shell(model)
    elif mode == 're-ckp': # revisit checkpoints
        metrics_trains, metrics_ins, metrics_out = [], [], []
        for model_path in model.get_all_checkpoints(config.dir_model_evaluate):
            model.restore_session(model_path)
            metrics_trains.append(model.run_evaluate(train, 0))
            metrics_ins.append(model.run_evaluate(test_in, 0))
            metrics_out.append(model.run_evaluate(test_out, 0))

        model.logger.info('Save metrics for all checkpoints...')
        pickle_save(metrics_trains, os.path.join(config.dir_output, 'metrics_trains.pickle'))
        pickle_save(metrics_ins, os.path.join(config.dir_output, 'metrics_ins.pickle'))
        pickle_save(metrics_out, os.path.join(config.dir_output, 'metrics_out.pickle'))
    elif mode == 'graph':
        metrics_trains = pickle_load(os.path.join(config.dir_output, 'metrics_trains.pickle'))
        metrics_ins = pickle_load(os.path.join(config.dir_output, 'metrics_ins.pickle'))
        metrics_out = pickle_load(os.path.join(config.dir_output, 'metrics_out.pickle'))

        if len(metrics_trains) != len(metrics_ins) or len(metrics_ins) != len(metrics_out):
            raise Exception('Unmatched number of metrics for trains/in/out')

        token_accs = np.zeros([3, len(metrics_trains)])
        f1s = np.zeros_like(token_accs)
        for i in range(len(metrics_trains)):
            token_accs[:, i] = [metrics_trains[i]['token_acc'], metrics_ins[i]['token_acc'], metrics_out[i]['token_acc']]
            f1s[:, i] = [metrics_trains[i]['f1'], metrics_ins[i]['f1'], metrics_out[i]['f1']]

        model.logger.info('Plotting ACC graph...')
        plt.clf()
        plt.legend(plt.plot(token_accs.T, marker='.'), ['train', 'in', 'out'])
        plt.savefig(os.path.join(config.dir_output, 'token_accs.png'), bbox_inches='tight')
        model.logger.info('Plotting F1 graph...')
        plt.clf()
        plt.legend(plt.plot(f1s.T, marker='.'), ['train', 'in', 'out'])
        plt.savefig(os.path.join(config.dir_output, 'f1s.png'), bbox_inches='tight')
    elif mode == 'multi':
        model.restore_latest_session(config.dir_model_evaluate)

        for i in range(len(config.filename_test_ins)):
            save_result_batchmode(model, config.filename_test_ins[i], i)
            save_result_batchmode(model, config.filename_test_outs[i], i)
            # save_result_batchmode(model, config.filename_test_outs[i], i)
    else:
        model.restore_latest_session(config.dir_model_evaluate)

        # evaluate
        model.evaluate(test_in, 0)
        model.evaluate(test_out, 0)

        '''cmkeyboard_labeled_folder = os.path.join(config.dir_data, 'test/CMKeyboard_Labeled')
        if not os.path.isdir(cmkeyboard_labeled_folder):
            cmkeyboard_labeled_folder = 'data/CMKeyboard_Labeled'
        for filename in sorted(glob.glob(os.path.join(cmkeyboard_labeled_folder, '*.txt'))):
            save_result_batchmode(model, filename, 0)'''

        # save_result_batchmode(config, model, config.filename_1p2gb_sample)

if __name__ == "__main__":
    buildModel()
