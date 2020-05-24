import features as F
from tqdm import tqdm

def flatten_predictions(preds):
    """ Flattens an output of cross validation. """

    return [lbl for fold in preds for conv in fold for lbl in conv]

def cross_validate(kFolds, model, features_fn, labels_fn, flatten=True):
    """ Cross validates a model with given features and labels. """

    test_preds, test_labels = [], []
    train_preds, train_labels = [], []

    for (train_dfs, test_dfs) in tqdm(kFolds):

        # Get a list of conversations
        conversation_list_train = [list(df.index) for df in train_dfs]
        conversation_list_test = [list(df.index) for df in test_dfs]

        # Construct CRF datasets
        X_train = [F.conversation2features(s, features_fn) for s in conversation_list_train]
        y_train = [F.conversation2labels(s, labels_fn) for s in conversation_list_train]

        X_test = [F.conversation2features(s, features_fn) for s in conversation_list_test]
        y_test = [F.conversation2labels(s, labels_fn) for s in conversation_list_test]

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
