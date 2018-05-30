import numpy as np

from .data_utils import get_all_chunks, get_pr, get_ordered_keys, NONE


class MetricCalc:
    def __init__(self, vocab_domains, vocab_intents, vocab_tags, vocab_chunk_types):
        self.vocab_domains = vocab_domains
        self.vocab_intents = vocab_intents
        self.vocab_tags = vocab_tags
        self.vocab_chunk_types = vocab_chunk_types

        self.ndomains = len(self.vocab_domains)
        self.nintents = len(self.vocab_intents)
        self.ntags = len(self.vocab_tags)
        self.nchunk_types = len(self.vocab_chunk_types)

        self.domain_conmat = np.zeros([self.ndomains, self.ndomains], dtype=np.int)
        self.intent_conmat = np.zeros([self.nintents, self.nintents], dtype=np.int)
        self.token_tag_conmat = np.zeros([self.ntags, self.ntags], dtype=np.int)
        self.correct_preds, self.total_correct, self.total_preds = 0., 0., 0.
        self.chunk_metrics = np.zeros([self.nchunk_types, 3], dtype=np.int)


    def add_batch(self, domains, domains_pred, intents, intents_pred, labels, labels_pred, sequence_lengths):
        for domain, domain_pred in zip(domains, domains_pred):
            self.domain_conmat[domain, domain_pred] += 1
        for intent, intent_pred in zip(intents, intents_pred):
            try:
                self.intent_conmat[intent, intent_pred] += 1
            except:
                print(str(intent) + "," + str(intent_pred))

        if len(labels) > 0 and type(labels[0][0]) is not int:
            labels = self.txt_label_to_idx(labels)
            labels_pred = self.txt_label_to_idx(labels_pred)

        for lab, lab_pred, length in zip(labels, labels_pred,
                                         sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            for (a, b) in zip(lab, lab_pred):
                self.token_tag_conmat[a, b] += 1

            lab_chunks = set(get_all_chunks(lab, self.vocab_tags))
            lab_pred_chunks = set(get_all_chunks(lab_pred,
                                                 self.vocab_tags))
            lab_correct_preds = lab_chunks & lab_pred_chunks

            for chunk_name, chunk_index in self.vocab_chunk_types.items():
                self.chunk_metrics[chunk_index, :] += np.array([
                    len([c for c in lab_correct_preds if c[0] == chunk_name]),
                    len([c for c in lab_pred_chunks if c[0] == chunk_name]),
                    len([c for c in lab_chunks if c[0] == chunk_name])
                ])
            self.correct_preds += len([c for c in lab_correct_preds if c[0] != NONE])
            self.total_preds += len([c for c in lab_pred_chunks if c[0] != NONE])
            self.total_correct += len([c for c in lab_chunks if c[0] != NONE])


    def txt_label_to_idx(self, labels):
        labels = np.array(labels)
        s = labels.shape
        return np.fromiter((self.vocab_tags[x] for x in labels.flatten()), np.int).reshape(s)


    def add(self, domain, domain_pred, intent, intent_pred, labels, labels_pred):
        self.add_batch([domain], [domain_pred], [intent], [intent_pred], [labels], [labels_pred], [len(labels)])


    def append_metric_for_conmat(self, metrics, conmat, key, classes):
        metrics[key+'_confusion'] = (classes, classes, conmat)

        pr_metrics = np.array([
            np.diagonal(conmat),
            np.sum(conmat, axis=0),
            np.sum(conmat, axis=1)
        ], dtype=np.float32).transpose()
        pr_metrics = get_pr(pr_metrics)
        metrics[key+'_pr'] = (classes, ["P", "R", "F1"], 100*pr_metrics)

        metrics[key+'_acc'] = 100*(np.trace(conmat, dtype=np.float32) / np.sum(conmat))  # same as micro-F1
        f1 = np.sum(pr_metrics[:, -1], dtype=np.float32) / np.count_nonzero(np.sum(conmat, axis=1))  # macro-F1
        metrics[key+'_f1'] = 100*f1

        return f1


    def get_metrics(self):
        metrics = {}

        self.append_metric_for_conmat(metrics, self.token_tag_conmat, 'tag', get_ordered_keys(self.vocab_tags))

        chunk_pr = np.vstack([
            self.chunk_metrics,
            np.array([self.correct_preds, self.total_preds, self.total_correct]).reshape(1, -1)
        ])
        chunk_pr = get_pr(chunk_pr)
        metrics['chunk_pr'] = (get_ordered_keys(self.vocab_chunk_types) + ["chunk_all_except_O"],
                               ["P", "R", "F1"],
                               100*chunk_pr)
        chunk_f1 = chunk_pr[-1, -1]
        metrics["chunk_f1"] = 100*chunk_f1

        f1 = chunk_f1
        if self.ndomains > 1:
            f1 += self.append_metric_for_conmat(metrics, self.domain_conmat, "domain",
                                                get_ordered_keys(self.vocab_domains))
        if self.nintents > 1:
            f1 += self.append_metric_for_conmat(metrics, self.intent_conmat, "intent",
                                                get_ordered_keys(self.vocab_intents))
        metrics["f1"] = 100*f1

        return metrics


# python -m model.metrics_calc
if __name__ == "__main__":
    from .config import Config
    config = Config()
    metric_calc = MetricCalc(config.vocab_tasks, config.vocab_tasks_intents[0],
                             config.vocab_tasks_tags[0], config.vocab_tasks_chunk_types[0])

    labels = [['O', 'O', 'B-MOVIE', 'I-MOVIE', 'I-MOVIE', 'O'],
              ['O', 'B-MOVIE', 'I-MOVIE', 'O']]

    labels_pred = [['O', 'O', 'B-MOVIE', 'I-MOVIE', 'I-MOVIE', 'O'],
                   ['O', 'O', 'I-MOVIE', 'O']]

    for lab, lab_pred in zip(labels, labels_pred):
        metric_calc.add(lab, lab_pred)

    import pprint
    pprint.pprint(metric_calc.get_metrics())