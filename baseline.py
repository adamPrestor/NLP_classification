import pandas as pd
import numpy as np

from hmmlearn import hmm
from sklearn.metrics import confusion_matrix


def calculate_matrix(l: list, tags_in: list, tags_out: list):
    transmat = []
    for tag in tags_in:
        tag_transitions = [x for x in l if x[0] is tag]
        n = len(tag_transitions)
        if n:
            trans = []
            for tag1 in tags_out:
                trans.append(len([x for x in tag_transitions if x[1] is tag1]))
            trans = [x / n for x in trans]
        else:
            trans = [0 for _ in range(len(tags_in) + 1)]
        transmat.append(trans)
    return transmat


def predict_naive(l, tags_in, tags_out):
    prediction_model = {}
    transmat = calculate_matrix(l, tags_in, tags_out)
    for i, values in enumerate(transmat):
        prediction_model[tags_in[i]] = tags_out[np.argmax(values)]
    return prediction_model


def test_naive(model: dict, l: list):
    return [model[el[0]] for el in l]


def evaluate_solution(pred: list, sol: list, tags: list, majority_class: str = 'C', verbose=True):
    confusion_mat = confusion_matrix(sol, pred, labels=tags)
    majority_i = tags.index(majority_class)

    n = len(pred)
    ca = sum([confusion_mat[i][i] for i in range(len(confusion_mat))]) / n

    if verbose:
        print(tags)
        print(confusion_mat)
        print(f'Our CA: {ca}')

        # print(np.sum(confusion_mat, axis=1))
        print(f'Majority CA: {np.sum(confusion_mat, axis=1)[majority_i] / n}')

    return confusion_mat


def prepare_transitions(l: list):
    """
    Adds the start and end tokens to the categories
    :param l:
    :return:
    """
    transitions = [item for sublist in
                   [list(zip(['START'] + list(el['CategoryBroad']), list(el['CategoryBroad']) + ['END'])) for el in l]
                   for item in sublist]
    return transitions
