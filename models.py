import os
import numpy as np
import pycrfsuite

class MajorityModel:
    """ Majority model. """

    def __init__(self):
        self.maj_class = None

    def train(self, X, y):
        y = [label for conversation in y for label in conversation]

        values, counts = np.unique(y, return_counts=True)
        i = np.argmax(counts)
        self.maj_class = values[i]

    def predict(self, X):
        preds = [[self.maj_class for feats in conv] for conv in X]

        return preds

class SklearnModel:
    """ Sklearn model wrapper. """

    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        X = [features for conversation in X for features, names in conversation]
        y = [label for conversation in y for label in conversation]

        self.model.fit(X, y)

    def predict(self, X):
        preds = []
        for conversation in X:
            X = [features for features, names in conversation]
            res = self.model.predict(X)
            preds.append(res)

        return preds

class CRFModel:
    """ CRF model wrapper. """

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_path = f'models/{self.model_name}.crfsuite'

    def parse_feats(self, X):
        def _to_dict(feats):
            features, names = feats
            d = {name:feat for feat, name in zip(features, names)}
            return d

        X = [[_to_dict(feats) for feats in conversation] for conversation in X]

        return X

    def train(self, X, y):
        X = self.parse_feats(X)

        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X, y):
            trainer.append(xseq, yseq)

        trainer.set_params({
            # include states and transitions that are possible, but not observed
            'feature.possible_states': True,
            'feature.possible_transitions': True
        })

        if not os.path.exists('models'):
            os.makedirs('models')

        # Train
        trainer.train(self.model_path)

    def predict(self, X):
        X = self.parse_feats(X)

        tagger = pycrfsuite.Tagger()
        tagger.open(self.model_path)

        preds = [tag for convo in X for tag in tagger.tag(convo)]

        tagger.close()

        return preds

