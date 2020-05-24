from datetime import timedelta
import numpy as np
from scipy import spatial
from nltk import ngrams


def build_ngram_model(messages, n=2):
    ngram_model = []
    for msg in messages:
        ngram_model.append(list(ngrams('_'.join(msg), n)))

    return ngram_model

def merge_feature_functions(fns):
    def _merged(message_id):
        all_features, all_names = [], []
        for fn in fns:
            feats, names = fn(message_id)
            all_features.extend(feats)
            all_names.extend(names)

        return all_features, all_names

    return _merged

# Only length as a feature
def wordcount(message_i, df):
    message = df.loc[message_i]['Message']

    count = len(message.split())

    features = [count]
    names = ['words']
    return features, names

# BoW as features
def message_bow(message_i, bow_values):
    #     message = df.loc[message_i]
    message_bow = bow_values[message_i]

    features = []
    for i, w in enumerate(message_bow):
        features.append(w)

    names = [f'bow_{i:03d}' for i in range(len(features))]

    return features, names


def book_context_words(message_i, df, messages, books_context, count=False, data_type='Book ID'):
    book_id = df.loc[message_i][data_type]
    if book_id not in books_context.keys():
        return [0]

    book_tokens = books_context[book_id]
    message_tokens = messages[message_i]

    names = ['context_words']

    if not count:
        features = sum([1 if token in book_tokens else 0 for token in message_tokens])
        return features, names


    count = 0
    for token in book_tokens:
        if token in message_tokens:
            count += 1

    features = [count]
    return features, names


def book_context_dist1(message_i, df, bow_values, bow_books, data_type='Book ID'):
    names = ['context_cos_distance']

    book_id = df.loc[message_i][data_type]
    if book_id not in bow_books.keys():
        return [1]

    message_bow = bow_values[message_i]
    book_bow = bow_books[book_id]

    if sum(message_bow) == 0:
        return [1], names

    dist = 1 - spatial.distance.cosine(message_bow, book_bow)
    return [dist], names

def book_context_dist2(message_i, df, bow_values, bow_books, data_type='Book ID'):
    names = ['book_similarity']

    book_id = df.loc[message_i][data_type]
    if book_id not in bow_books.keys():
        features = [0]
        return features, names

    message_bow = bow_values[message_i]
    book_bow = bow_books[book_id]

    features = [np.dot(message_bow, book_bow)]

    return features, names


def length(message_i, df):
    message = df.loc[message_i]['Message']

    count = len(message)

    features = [count]
    names = ['length']

    return features, names


def previous_msg(message_i, bow_values):
    names = ['prev_msg_dist']

    tokens = bow_values[message_i]
    tokens_p = bow_values[message_i - 1]

    if not sum(tokens) or not sum(tokens_p):
        return [0], names

    features = [np.dot(tokens, tokens_p)]
    return features, names

# Sentiment
def sentiment(message_id, messages, sentiment_analysis, normalize=True):
    message = messages[message_id]
    sent = sentiment_analysis.sentiment(message, normalize=normalize)

    names = ['sentiment']

    return [sent], names

def recent_activity(message_id, df):
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


    features = [n_last_5min, n_posts_5min, n_users_5min]
    names = ['recent_user_posts', 'recent_posts', 'recent_users']

    return features, names

def get_label(message_i, df):
    message = df.loc[message_i]
    return message['CategoryBroad']


def conversation2features(conversation, feature_fn):
    features = [feature_fn(msg_i) for msg_i in conversation]
    return features

def conversation2labels(conversation, labels_fn):
    labels = [labels_fn(msg_i) for msg_i in conversation]
    return labels
