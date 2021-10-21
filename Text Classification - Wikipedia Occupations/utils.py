"""
This file provides some basic function to extracting occupations.
There is no need to modify this file unless you want.
"""

import gzip
import json
import nltk
from tqdm import tqdm
import numpy as np
from keras.preprocessing.sequence import pad_sequences


class InputSample(object):
    def __init__(self, title, summary, occupation):
        self.title = title
        self.summary = summary
        self.occupation = occupation


def pad_sentence(sentence, max_len):
    '''
    make all sentences have the same length
    :param sentence:
    :param max_len:
    :return:
    '''
    seg_id = pad_sequences([sentence], maxlen=max_len, padding='post')
    return seg_id[0]


def get_label():
    occupations = [
        'yago:Politician',
        'yago:Researcher',
        'yago:Football_player',
        'yago:Writer',
        'yago:Actor',
        'yago:Painter',
        'yago:Journalist',
        'yago:University_teacher',
        'yago:Singer',
        'yago:Poet',
        'yago:Composer',
        'yago:Military_personnel',
        'yago:Lawyer',
        'yago:Film_actor',
        'yago:Businessperson',
        'yago:Historian',
        'yago:Musician',
        'yago:Film_director',
        'yago:Screenwriter',
        'yago:Physician'
    ]

    labels = {occ_id: index for index, occ_id in enumerate(occupations)}
    id_to_labels = {index: occ_id for index, occ_id in enumerate(occupations)}
    return labels, id_to_labels


def load_data(filename):
    '''
    load original data
    :param filename:
    :return:
    '''
    with gzip.open(filename, 'rt') as fp:
        for line in fp:
            people = json.loads(line)
            occ_key = 'occupations'
            occupations = people[occ_key] if occ_key in people else None
            sample = InputSample(people['title'], people['summary'], occupations)
            yield sample


def gen_vocabulary(data_file, vocab_file):
    '''
    generate a word list given an input corpus
    :param data_file:
    :param vocab_file:
    :return:
    '''
    vocab = set()
    for sample in tqdm(load_data(data_file)):
        sentence = str.lower(sample.summary)
        tokens = nltk.word_tokenize(sentence)
        vocab.update(set(tokens))

    with open(vocab_file, 'w', encoding='utf8')as f:
        f.write('\n'.join(list(vocab)))

    print('done! The size of vocabulary is {a}.'.format(a=len(vocab)))


def load_vocabulary(vocab_file):
    '''
    load vocabulary and create an id for each token.
    <pad> means padding token, <unk> means unknown token
    :param vocab_file:
    :return:
    '''
    vocab_to_id = dict()
    with open(vocab_file, encoding='utf8')as f:
        words = f.readlines()
        for w_id, word in enumerate(words):
            word = word.replace('\n', '')
            vocab_to_id[word] = w_id+1
    vocab_to_id['<pad>'] = 0
    vocab_to_id['<unk>'] = len(vocab_to_id)
    return vocab_to_id


def read_dataset(data_file, vocab_to_id, sent_len, debug=False):
    '''
    read training set or test set
    :param data_file:
    :param vocab_to_id:
    :param sent_len: the
    :param debug: load only a small fraction of samples to debug
    :return: model's input and labels
    need about 1min31s for training set and 2min for test set
    '''

    labels, _ = get_label()
    unknown_id = len(vocab_to_id) - 1
    data_x, data_y = list(), list()
    cnt = 0

    for sample in tqdm(load_data(data_file)):

        # for debugging
        cnt += 1
        if debug and cnt > 100:
            break

        summary = str.lower(sample.summary)
        tokens = nltk.word_tokenize(summary)
        token_ids = [vocab_to_id.get(t, unknown_id) for t in tokens]
        token_ids = pad_sentence(token_ids, sent_len)
        data_x.append(token_ids)
        occupations = sample.occupation

        # train
        if occupations:
            y_vector = [1 if label in occupations else 0 for label in labels]
            data_y.append(y_vector)
        # test
        else:
            data_y.append(0)

    return np.array(data_x), np.array(data_y)


def f1_score(true_labels, pred_labels):
    """Compute the F1 score."""
    nb_correct, nb_pred, nb_true = 0, 0, 0
    for true, pred in zip(true_labels, pred_labels):
        nb_correct += len(true & pred)
        nb_pred += len(pred)
        nb_true += len(true)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score, p, r

