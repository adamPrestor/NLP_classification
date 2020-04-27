import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm
from sklearn.metrics import confusion_matrix

# delete later, it's here for the debug
from baseline import *


def csv_parse(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path, delimiter=';')


def split_train_test(df: pd.DataFrame):
    """
    Splits the dataframe by School, Bookclub and Topic, then divides 80% of conversations into train set and 20%
    into test set.
    :param df:
    :return: [(train, test)] - both are lists of dataframes
    """
    conversations = df.groupby(by=['School', 'Bookclub', 'Topic'])
    schools = df['School'].unique()
    output = []
    for excluded_school in schools:
        train = []
        test = []
        for x in conversations:
            if not x[0][0] is excluded_school:
                train.append(x[1])
            else:
                test.append(x[1])
        output.append((train, test))
    return output


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def get_statistical_feeds(df):
    print(df.columns)
    df1 = pd.crosstab(df['Bookclub'], df['CategoryBroad'], normalize=True)
    print(df1)
    df2 = pd.crosstab(df['CategoryBroad'], df['School'])  # , normalize=True)
    print(df2)
    df4 = pd.crosstab(df['Topic'], df['Bookclub'], normalize=True)
    print(df4.values)
    df5 = df['CategoryBroad'].value_counts()
    print(df5)

    set = df['Message'].str.len()
    set.reindex(df['CategoryBroad'].values)

    df['MessageLength'] = df['Message'].apply(str).apply(len)

    df6 = df.groupby('CategoryBroad')['MessageLength'].mean()
    print(df6)

    # msg_list = list(df['Message'])

    return df2, df5, df6


def output_plots(data: pd.DataFrame, title, y_label, filename):
    labels = [name[:10] for name in data.columns]
    row_labels = data.index

    print(labels, '\n', row_labels)

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    reacts = []
    x = np.arange(len(labels))
    n = len(labels)
    width = 0.15

    for i, row in enumerate(data.index):
        reacts.append(ax.bar(x - (n // 2 - i) * width, data.loc[row, :], width, label=row))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for react in reacts:
        autolabel(react, ax)

    fig.tight_layout()

    plt.savefig(filename)
    plt.show()


def output_single(data: pd.Series, title, y_label, filename):
    labels = data.index

    fig, ax = plt.subplots()
    x = np.arange(len(labels))

    react = ax.bar(x, data.values, 0.7)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(react, ax)

    fig.tight_layout()

    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    df = csv_parse('input.csv')

    # df1, df2, df3 = get_statistical_feeds(df)

    # output_plots(df1, 'CategoryBroad by school', 'Counts', 'broad-school')
    # output_single(df2, 'CategoryBroad in dataset distribution', 'Counts', 'broad')
    # output_single(df3, 'Message length by CategoryBroad', 'Length', 'length')

    unique_tags = list(df['CategoryBroad'].unique())

    split = split_train_test(df)

    train, test = split[0]

    train_t = prepare_transitions(train)
    test_t = prepare_transitions(test)

    print(len(train_t))

    model = predict_naive(train_t, ['START'] + unique_tags, unique_tags + ['END'])

    prediction = test_naive(model, test_t)

    print(model)
    print(prediction)

    evaluate_solution(prediction, [x[1] for x in test_t], unique_tags + ['END'])

    # model = hmm.GaussianHMM(n_components=len(unique_tags) + 1, covariance_type='full')
    # model.startprob_ = start_prob
    # model.transmat_ = transmat
