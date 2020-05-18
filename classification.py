import os
import random
import functools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

from preprocessing import read_dataset
from preprocessing import Tokenization, StopWordsRemover, Lemmatization, RoofRemoval, SpellingCorrection
from preprocessing import GibberishDetector, TokenGrouping, TokenDictionary
import pycrfsuite

from baseline import evaluate_solution
from csv_parser import split_train_test
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix


# 1. Define feature functions

# Only length as a feature
def get_features_length(message_i):
    message = df.loc[message_i]['Message']
    feature_list = []

    count = len(message.split())
    feature_list.append(count)

    return feature_list


# BoW as features
def get_features_bow(message_i, bow_values):
    #     message = df.loc[message_i]
    message_bow = bow_values[message_i]

    features = []
    for i, w in enumerate(message_bow):
        features.append(w)

    return features


def get_label(message_i):
    message = df.loc[message_i]
    return message['CategoryBroad']


def conversation2features(conversation, feature_fn):
    features = [feature_fn(msg_i) for msg_i in conversation]

    return features


def conversation2labels(conversation, labels_fn):
    labels = [labels_fn(msg_i) for msg_i in conversation]
    return labels


if __name__ == "__main__":
    dataset_path = 'input.csv'
    df = read_dataset(dataset_path)

    tokenizer = Tokenization()
    stop_words_remover = StopWordsRemover('data/stopwords-sl-custom.txt')
    lemmatizer = Lemmatization()

    roof_removal = RoofRemoval()\

    spelling_correction = SpellingCorrection('data/dict-sl.txt', roof_removal)

    gibberish_detector = GibberishDetector(roof_removal)
    # Train gibberish_detector
    gibberish_detector.train('data/dict-sl.txt', 'data/gibberish_good.txt', 'data/gibberish_bad.txt')

    token_grouping = TokenGrouping(gibberish_detector)

    # Tokenization
    messages = df.Message
    messages = [tokenizer.tokenize(message) for message in messages]

    # Remove stop words
    messages = [stop_words_remover.remove_stopwords(tokens) for tokens in messages]

    # Lemmatization
    messages = [[lemmatizer.lemmatize(token) for token in message] for message in messages]

    # Roof removal
    messages = [[roof_removal.remove(token) for token in message] for message in messages]

    # Token grouping
    messages = [[token_grouping.group_tokens(token) for token in message] for message in messages]

    # Create BoW dictionary
    token_dict = TokenDictionary(messages)

    # Get tf-idf weighted BoW representations
    bow = np.stack([token_dict.bag_of_words(message) for message in messages])
    bow_tfidf = np.stack([token_dict.bag_of_words(message, tf_idf=True) for message in messages])

    kFolds = split_train_test(df)

    combine_matrix_test = np.zeros([len(list(df.CategoryBroad.unique())), len(list(df.CategoryBroad.unique()))], dtype=np.uint32)
    combine_matrix_train = np.zeros([len(list(df.CategoryBroad.unique())), len(list(df.CategoryBroad.unique()))], dtype=np.uint32)

    for (train_dfs, test_dfs) in kFolds:
        print(test_dfs[0]['School'].unique())
        train_dfs_all = pd.concat(train_dfs)
        test_dfs_all = pd.concat(test_dfs)
        majority_class = train_dfs_all.CategoryBroad.value_counts().index[0]

        conversation_list_train = list(train_dfs_all.index)
        conversation_list_test = list(test_dfs_all.index)

        # features_fn = get_features_length
        # features_fn = functools.partial(get_features_bow, bow_values=bow)
        features_fn = functools.partial(get_features_bow, bow_values=bow_tfidf)
        labels_fn = get_label

        X_train = [features_fn(s)for s in conversation_list_train]
        y_train = [labels_fn(s) for s in conversation_list_train]

        X_test = [features_fn(s) for s in conversation_list_test]
        y_test = [labels_fn(s) for s in conversation_list_test]

        # clf = RandomForestClassifier(n_estimators=100)
        # clf = GaussianNB()
        # clf = DecisionTreeClassifier()
        clf = KNeighborsClassifier(metric='minkowski', n_neighbors=9)
        # clf = LinearSVC()

        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        labels = [tag for convo in y_test for tag in convo]
        tags = list(df.CategoryBroad.unique())

        conf_mat = evaluate_solution(preds, labels, tags, majority_class=majority_class, verbose=False)
        combine_matrix_test = combine_matrix_test + conf_mat

        preds = clf.predict(X_train)
        labels = [tag for convo in y_train for tag in convo]
        tags = list(df.CategoryBroad.unique())

        conf_mat = evaluate_solution(preds, labels, tags, majority_class=majority_class, verbose=False)
        combine_matrix_train = combine_matrix_train + conf_mat

    print("Test results ======================")
    print(combine_matrix_test)
    print(f'Combined CA: {sum([combine_matrix_test[i][i] for i in range(len(combine_matrix_test))]) / np.sum(combine_matrix_test)}')
    pretty_plot_confusion_matrix(pd.DataFrame(combine_matrix_test, columns=tags, index=tags), cmap="Blues")

    print("Train results ======================")
    print(combine_matrix_train)
    print(f'Combined CA: {sum([combine_matrix_train[i][i] for i in range(len(combine_matrix_train))]) / np.sum(combine_matrix_train)}')
    pretty_plot_confusion_matrix(pd.DataFrame(combine_matrix_train, columns=tags, index=tags), cmap="Blues")

    plt.show()
