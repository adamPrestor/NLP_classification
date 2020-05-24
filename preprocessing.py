import re
import string
from collections import Counter

import pandas as pd
import numpy as np
from nltk.tokenize.casual import TweetTokenizer
import lemmagen
from lemmagen.lemmatizer import Lemmatizer
import difflib

def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path, delimiter=';')
    df['CategoryBroad'] = pd.Categorical(df['CategoryBroad'])
    df.Message = df.Message.fillna('')

    return df

class Tokenization():
    """ Tokenization module. """

    def __init__(self):
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)

    def split_punc(self, tokens):
        """  Split punctuation without space ('neki.halo' -> 'neki', '.', 'halo') """
        tokens_out = []
        for token in tokens:
            match = re.match('(\w+)\.(\w+)', token)
            if match is not None:
                l,r = match.groups()
                tokens_out.extend([l, '.', r])
            else:
                tokens_out.append(token)

        return tokens_out

    def split_num(self, tokens):
        """ Split numbers to separate tokens ('username15halo' -> 'username', '15', 'halo') """
        tokens_out = []
        for token in tokens:
            new_tokens = re.findall('\d+|\D+', token)
            tokens_out.extend(new_tokens)

        return tokens_out

    def tokenize(self, message):
        """ Tokenizes message. """

        tokens = self.tokenizer.tokenize(message)

        # Fix punctuations
        tokens = self.split_punc(tokens)

        # Separate numbers
        tokens = self.split_num(tokens)

        return tokens

class StopWordsRemover():
    """ Stop words remover. Works on a given stop word txt file. """

    def __init__(self, stop_words_txt):
        with open(stop_words_txt, 'r') as file:
            self.stopwords = {line.strip() for line in file}

    def remove_stopwords(self, tokens):
        filtered = [token for token in tokens if token not in self.stopwords]
        return filtered


class Lemmatization():
    def __init__(self):
        self.lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)

    def lemmatize(self, token):
          return self.lemmatizer.lemmatize(token)


class RoofRemoval():
    """ Replaces slovenian special symbols. """

    def __init__(self):
        self.replacements = dict([('ž', 'z'),('č', 'c'),('š', 's'),('ć', 'c'),('đ', 'dz')])
        self.pattern = re.compile("|".join(self.replacements.keys()))

    def remove(self, token):
        token = self.pattern.sub(lambda x: self.replacements[x.group(0)], token)
        return token


class SpellingCorrection():
    """ Corrects misspelled words by replacing them with most similar word from dict. """

    def __init__(self, dict_txt, roof_removal=None):

        with open(dict_txt, 'r') as file:
            dictionary = (line.strip() for line in file)

            if roof_removal is not None:
                dictionary = (roof_removal.remove(w) for w in dictionary)

            self.dictionary = list(dictionary)

        self.roof_removal = roof_removal

    def find_close(self, token, n=1, cutoff=0.9):
        close_matches = difflib.get_close_matches(token, self.dictionary, n=n, cutoff=cutoff)

        scores = [difflib.SequenceMatcher(None, token, w).ratio() for w in close_matches]

        return list(zip(close_matches, scores))

    def replace_if_close(self, token, thresh=0.9):
        res = self.find_close(token, n=1, cutoff=thresh)
        if len(res) < 1:
            return token

        word, sim = res[0]
        if sim < 1:
            print(f'{token} -> {word} ({sim})')

        return word


class GibberishDetector():
    def __init__(self, roof_removal):
        """Gibberish detector trained on a dictionary of real words (using Markov chains)."""

        self.invalid_regex = re.compile('[^a-z]')

        self.states = string.ascii_lowercase + '*^$'
        self.state_index = {char: i for i, char in enumerate(self.states)}

        self.probs = np.zeros((len(self.states), len(self.states)))
        self.threshold = 0.0

        self.roof_removal = roof_removal


    def normalize(self, string):
        """Replace invalid characters (non-alphabet) with *"""

        # Remove roofs
        string = self.roof_removal.remove(string)

        # Replace out-of-dict chars
        string = self.invalid_regex.sub('*', string)

        # Surround with ^ and $
        string = '^' + string + '$'

        return string

    def ngram(self, string, n):
        """ Return all n grams from string"""

        filtered = self.normalize(string)
        for start in range(0, len(filtered) - n + 1):
            yield ''.join(filtered[start:start + n])

    def train(self, dictionary_txt, good_txt, bad_txt):
        counts = np.zeros((len(self.states), len(self.states)))

        with open(dictionary_txt, 'r') as file:
            word_iter = (self.normalize(line.strip()) for line in file)

            for word in word_iter:
                for c1, c2 in self.ngram(word, 2):
                    c1i = self.state_index[c1]
                    c2i = self.state_index[c2]

                    counts[c1i, c2i] += 1


        # Add small probability even to missing transitions
        laplace_vector = np.maximum((counts.sum(axis=1)*0.01/len(counts)), 1)[:, np.newaxis]
        counts = np.maximum(counts, laplace_vector)

        # Compute log probabilities
        sums = counts.sum(axis=1)[:, np.newaxis]
        self.probs = np.log(counts / sums)

        # Compute best threshold
        with open(good_txt, 'r') as file:
            word_iter = (line.strip() for line in file)
            good_probs = np.array([self.word_probability(word) for word in word_iter])

        with open(bad_txt, 'r') as file:
            word_iter = (line.strip() for line in file)
            bad_probs = np.array([self.word_probability(word) for word in word_iter])

        min_g = np.min(good_probs)
        max_b = np.max(bad_probs)

        self.threshold = (min_g + max_b) * 0.5

        # Test threshold
        print(f'Correct good: {np.mean(good_probs > self.threshold)}')
        print(f'Correct bad: {np.mean(bad_probs <= self.threshold)}')

    def word_probability(self, word):
        word = self.normalize(word)

        log_prob = 0.0
        count = 0
        for c1, c2 in self.ngram(word, 2):
            c1i = self.state_index[c1]
            c2i = self.state_index[c2]

            log_prob += self.probs[c1i, c2i]
            count += 1

        return np.exp(log_prob / count)

    def is_gibberish(self, word):
        return self.word_probability(word) <= self.threshold


class TokenGrouping():
    def __init__(self, gibberish_detector):
        self.gibberish_detector = gibberish_detector
        self.other_regex = re.compile(r'[^\w?!.,-]')

    def group_tokens(self, token, verbose=False):
        """ Groups some tokens into the same group (number, gibberish, ...). """

        if token.isdigit():
            msg = f'<number> <- {token}'
            token = '<number>'
        elif len(token) > 4 and self.gibberish_detector.is_gibberish(token):
            msg = f'<gibberish> <- {token}'
            token = '<gibberish>'
        elif self.other_regex.search(token) is not None:
            msg = f'<other> <- {token}'
            token = '<other>'

        if verbose and msg is not None:
            print(msg)

        return token



class TokenDictionary():
    """ Generates token dictionary. Can convert tokens to BoW representation. """

    def __init__(self, documents, dict_size=512):
        self.dict_size = dict_size

        all_tokens = [token for document in documents for token in document]

        # Find most common tokens and construct a dict from them
        cnt = Counter(all_tokens)
        most_common = cnt.most_common(dict_size)

        # Out-of-dict token
        remaining = len(all_tokens) - sum(count for _, count in most_common)
        most_common.append(('<OOD>', remaining))

        token_dictionary = [token for token, count in most_common]
        self.token_map = {token: i for i, token in enumerate(token_dictionary)}

        # Compute idf for each word
        bow = np.stack([self.bag_of_words(document) for document in documents])
        num_documents = len(documents)
        word_occurences = np.sum(bow>0, axis=0)

        self.idf = np.log(num_documents / word_occurences)

    def get_token(self, token):
        """ Get in-dict token for a given token. """

        if token not in self.token_map:
            return '<OOD>'

        return token

    def bag_of_words(self, tokens, tf_idf=False, relative=False, include_ood=False):
        """ Convert a list of tokens to a bag-of-words representation. """

        bow = np.zeros(self.dict_size + 1)
        if len(tokens) == 0:
            return bow

        # Absolute freqs
        for token in tokens:
            token = self.get_token(token)
            i = self.token_map[token]
            bow[i] += 1

        # Relative freqs
        if relative or tf_idf:
            bow = bow / len(tokens)

        # TF-IDF
        if tf_idf:
            tf = np.log(1 + bow)
            idf = self.idf

            bow = tf * idf

        if not include_ood:
            bow = bow[:-1]

        return bow


class SentimentAnalysis:
    """ Sentiment analysis from a dictionary of positive and negative words. """

    def __init__(self, negative_words_file, positive_words_file, roof_removal):
        with open(negative_words_file) as file:
            self.negative_words = {roof_removal.remove(word.strip()) for word in file}

        with open(positive_words_file) as file:
            self.positive_words = {roof_removal.remove(word.strip()) for word in file}

    def sentiment(self, message, normalize=True):
        sent = 0
        for token in message:
            if token in self.negative_words:
                sent -= 1
            elif token in self.positive_words:
                sent += 1

        # Normalize (-1, 1)
        if sent != 0 and normalize:
            sent = sent / len(message)

        return sent
