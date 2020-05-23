import io
import os
import random
import functools
from itertools import chain

import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import seaborn as sn

from nltk import ngrams

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


def get_features_context(message_i, books_context, count=False, data_type='Book ID'):
    book_id = df.loc[message_i][data_type]
    if book_id not in books_context.keys():
        return [0]

    book_tokens = books_context[book_id]
    message_tokens = messages[message_i]

    if not count:
        return [sum([1 if token in book_tokens else 0 for token in message_tokens])]

    count = 0
    for token in book_tokens:
        if token in message_tokens:
            count += 1
    return [count]


def get_cosine_distance(message_i, bow_values, bow_books, data_type='Book ID'):
    book_id = df.loc[message_i][data_type]
    if book_id not in bow_books.keys():
        return [1]

    message_bow = bow_values[message_i]
    book_bow = bow_books[book_id]

    if sum(message_bow) == 0:
        return [1]

    return [1 - spatial.distance.cosine(message_bow, book_bow)]


def get_tfidf_from_book(message_i, bow_values, bow_books, data_type='Book ID'):
    book_id = df.loc[message_i][data_type]
    if book_id not in bow_books.keys():
        return [0]

    message_bow = bow_values[message_i]
    book_bow = bow_books[book_id]

    return [np.dot(message_bow, book_bow)]


def get_features_rawlength(message_i):
    message = df.loc[message_i]['Message']
    feature_list = []

    count = len(message)
    feature_list.append(count)

    return feature_list


def get_features_previous_msg(message_i, bow_values):
    tokens = bow_values[message_i]
    tokens_p = bow_values[message_i - 1]

    if not sum(tokens) or not sum(tokens_p):
        return [0]

    return [np.dot(tokens, tokens_p)]

# Sentiment
def get_features_sent(message_id):
    message = messages_sent[message_id]
    sent = sa.sentiment(message)

    features = {'sentiment': sent}

    return features

def get_features_history(message_id):
    entry = df.loc[message_id]
    time = entry['Message Time']
    username = entry['Name']

    conversation = df[df['School'] == entry['School']]
    conversation = conversation[conversation['Bookclub'] == entry['Bookclub']]
    conversation = conversation[conversation['Topic'] == entry['Topic']]

    time_mask = (conversation['Message Time'] < time) & (conversation['Message Time'] > time - timedelta(minutes=5))
    username_mask = conversation['Name'] == username

    n_last_5min = len(conversation[time_mask & username_mask])
    n_posts_5min = len(conversation[time_mask])
    n_users_5min = len(conversation[time_mask]['Name'].unique())

    features = {
        'recent_user_posts': n_last_5min,
        'recent_posts': n_posts_5min,
        'recent_users': n_users_5min
    }

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


def build_ngram_model(messages, n=2):
    ngram_model = []
    for msg in messages:
        ngram_model.append(list(ngrams('_'.join(msg), n)))

    return ngram_model


if __name__ == "__main__":
    dataset_path = 'data/discussion_data.csv'
    df = read_dataset(dataset_path)

    tokenizer = Tokenization()
    stop_words_remover = StopWordsRemover('data/stopwords-sl-custom.txt')
    lemmatizer = Lemmatization()

    roof_removal = RoofRemoval()
    spelling_correction = SpellingCorrection('data/dict-sl.txt', roof_removal)

    gibberish_detector = GibberishDetector(roof_removal)
    # Train gibberish_detector
    gibberish_detector.train('data/dict-sl.txt', 'data/gibberish_good.txt', 'data/gibberish_bad.txt')

    token_grouping = TokenGrouping(gibberish_detector)

    # tokenize the books
    book_tokens = {}
    for book in os.listdir('data/books'):
        book_id = int(book.split('.')[0])
        with io.open(os.path.join('data/books', book), mode='r', encoding='utf-8') as f:
            content = f.read()
            tokens = tokenizer.tokenize(content)
            tokens = stop_words_remover.remove_stopwords(tokens)
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            tokens = [roof_removal.remove(token) for token in tokens]
            tokens = [token_grouping.group_tokens(token) for token in tokens]
            # remove gibberish and punctuations
            tokens = [token for token in tokens if token.isalpha() and not token == '<other>']
            book_tokens[book_id] = tokens

    # tokenize the topics
    topics_tokens = {}
    for topic in df['Topic'].unique():
        tokens = tokenizer.tokenize(topic)
        tokens = stop_words_remover.remove_stopwords(tokens)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [roof_removal.remove(token) for token in tokens]
        tokens = [token_grouping.group_tokens(token) for token in tokens]
        # remove gibberish and punctuations
        tokens = [token for token in tokens if token.isalpha() and not token == '<other>']
        topics_tokens[topic] = tokens

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
    token_dict = TokenDictionary(messages +
                                 [cont for cont in book_tokens.values()] +
                                 [cont for cont in topics_tokens.values()], dict_size=1024)

    # Create BoW of ngrams
    ngram_arr = build_ngram_model([msg.split() for msg in df.Message], 2)
    ngram_dict = TokenDictionary(ngram_arr, dict_size=1024)

    # Get tf-idf weighted BoW representations
    bow = np.stack([token_dict.bag_of_words(message) for message in messages])
    bow_tfidf = np.stack([token_dict.bag_of_words(message, tf_idf=True) for message in messages])
    book_bow = dict([(key, token_dict.bag_of_words(value)) for (key, value) in book_tokens.items()])
    book_tfidf = dict([(key, token_dict.bag_of_words(value, tf_idf=True)) for (key, value) in book_tokens.items()])
    topic_bow = dict([(key, token_dict.bag_of_words(value)) for (key, value) in topics_tokens.items()])
    topic_tfidf = dict([(key, token_dict.bag_of_words(value, tf_idf=True)) for (key, value) in topics_tokens.items()])

    # Get ngram model
    ngram_model = np.stack([ngram_dict.bag_of_words(gram) for gram in ngram_arr])

    mask = df['Book ID'].isin(book_tokens.keys())
    df = df[mask]

    kFolds = split_train_test(df)

    combine_matrix_test = np.zeros([len(list(df.CategoryBroad.unique())), len(list(df.CategoryBroad.unique()))],
                                   dtype=np.uint32)
    combine_matrix_train = np.zeros([len(list(df.CategoryBroad.unique())), len(list(df.CategoryBroad.unique()))],
                                    dtype=np.uint32)

    for (train_dfs, test_dfs) in kFolds:
        print(test_dfs[0]['School'].unique())
        train_dfs_all = pd.concat(train_dfs)
        test_dfs_all = pd.concat(test_dfs)
        majority_class = train_dfs_all.CategoryBroad.value_counts().index[0]

        conversation_list_train = list(train_dfs_all.index)
        conversation_list_test = list(test_dfs_all.index)

        features_fns = []
        features_fns.append(get_features_length)
        features_fns.append(get_features_rawlength)
        features_fns.append(functools.partial(get_features_previous_msg, bow_values=bow))
        # features_fns.append(functools.partial(get_features_bow, bow_values=bow))
        features_fns.append(functools.partial(get_features_bow, bow_values=bow_tfidf))
        features_fns.append(functools.partial(get_features_bow, bow_values=ngram_model))
        features_fns.append(functools.partial(get_features_context, books_context=book_tokens, count=True))
        # features_fns.append(functools.partial(get_cosine_distance, bow_values=bow_tfidf, bow_books=book_tfidf))
        # features_fns.append(functools.partial(get_tfidf_from_book, bow_values=bow, bow_books=book_tfidf))
        features_fns.append(functools.partial(get_features_context, books_context=topics_tokens, count=True, data_type='Topic'))
        # features_fns.append(functools.partial(get_cosine_distance, bow_values=bow, bow_books=topic_tfidf, data_type='Topic'))
        # features_fns.append(functools.partial(get_tfidf_from_book, bow_values=bow, bow_books=topic_tfidf, data_type='Topic'))
        labels_fn = get_label

        X_train = [list(chain.from_iterable([features_fn(s) for features_fn in features_fns])) for s in conversation_list_train]
        y_train = [labels_fn(s) for s in conversation_list_train]

        X_test = [list(chain.from_iterable([features_fn(s) for features_fn in features_fns])) for s in conversation_list_test]
        y_test = [labels_fn(s) for s in conversation_list_test]

        # clf = RandomForestClassifier(n_estimators=500)
        # clf = GaussianNB()
        # clf = DecisionTreeClassifier()
        # clf = KNeighborsClassifier(metric='minkowski', n_neighbors=9)
        clf = LinearSVC(max_iter=4000)

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
    print(
        f'Combined CA: {sum([combine_matrix_test[i][i] for i in range(len(combine_matrix_test))]) / np.sum(combine_matrix_test)}')
    pretty_plot_confusion_matrix(pd.DataFrame(combine_matrix_test, columns=tags, index=tags), cmap="Blues")

    print("Train results ======================")
    print(combine_matrix_train)
    print(
        f'Combined CA: {sum([combine_matrix_train[i][i] for i in range(len(combine_matrix_train))]) / np.sum(combine_matrix_train)}')
    pretty_plot_confusion_matrix(pd.DataFrame(combine_matrix_train, columns=tags, index=tags), cmap="Blues")

    plt.show()
