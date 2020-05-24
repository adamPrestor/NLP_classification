import features as F
from tqdm import tqdm

def flatten_predictions(preds):
    """ Flattens an output of cross validation. """

    return [lbl for fold in preds for conv in fold for lbl in conv]


def cross_validate(folds_data, model, flatten=True):
    """ Cross validates a model with given kfolds. """

    test_preds, test_labels = [], []
    train_preds, train_labels = [], []

    for train_data, test_data in tqdm(folds_data):
        X_train, y_train = train_data
        X_test, y_test = test_data

        model.train(X_train, y_train)

        preds = model.predict(X_test)
        test_preds.append(preds)
        test_labels.append(y_test)

        preds = model.predict(X_train)
        train_preds.append(preds)
        train_labels.append(y_train)

    if flatten:
        res = [train_preds, train_labels, test_preds, test_labels]
        res = [flatten_predictions(pred) for pred in res]
        train_preds, train_labels, test_preds, test_labels = res

    return train_preds, train_labels, test_preds, test_labels
