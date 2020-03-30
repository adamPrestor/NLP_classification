import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def csv_parse(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, delimiter=';')

    return df


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

    set = data['Message'].str.len()
    set.reindex(data['CategoryBroad'].values)

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
        reacts.append(ax.bar(x - (n//2-i)*width, data.loc[row, :], width, label=row))

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


if __name__=='__main__':
    data = csv_parse('input.csv')

    df1, df2, df3 = get_statistical_feeds(data)

    # output_plots(df1, 'CategoryBroad by school', 'Counts', 'broad-school')
    # output_single(df2, 'CategoryBroad in dataset distribution', 'Counts', 'broad')
    output_single(df3, 'Message length by CategoryBroad', 'Length', 'length')
