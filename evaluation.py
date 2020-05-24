import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay

from confusion_matrix_pretty_print import pretty_plot_confusion_matrix, plot_confusion_matrix_from_data


def class_specific_string(class_results):
    return f"{str(class_results[0]):<6} " \
           f"{str(class_results[1]):<10} " \
           f"{str(class_results[2]):<10} " \
           f"{str(class_results[3]):<10}"


class Evaluator:

    def __init__(self, predictions, truth, tags):
        self.y_pred = predictions
        self.y_true = truth
        self.tags = tags

    def calculate_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred, labels=self.tags)

    def get_f1_measure(self, averaged=True):
        """ returns f1 measurement of the predictions """
        return self.get_measurement(f1_score, averaged)

    def get_precision(self, averaged=True):
        """ returns precision measurement of the predictions """
        return self.get_measurement(precision_score, averaged)

    def get_recall(self, averaged=True):
        """ returns recall measurement of the predictions """
        return self.get_measurement(recall_score, averaged)

    def get_accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def get_classification_report(self, output=None, plot=False):
        """ get full report with confusion matrix.
         F1, recall and precision are calculated for each class and weight averaged
         output tells us file path to output the results. if set to none, it outputs to console.
         """
        confusion_mat = self.calculate_confusion_matrix()
        ca = self.get_accuracy()
        precision = self.get_precision()
        recall = self.get_recall()
        f_measure = self.get_f1_measure()

        lines = [
            "Confusion matrix:",
            f"{pd.DataFrame(confusion_mat, index=self.tags, columns=self.tags)}",
            "",
            f"Accuracy : {ca:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall   : {recall:.4f}",
            f"F1 score : {f_measure:.4f}",
            "", "",
            class_specific_string(['class', 'precision', 'recall', 'f1 score']),
            "",
        ]

        precisions = self.get_precision(averaged=False)
        recalls = self.get_recall(averaged=False)
        f_measures = self.get_f1_measure(averaged=False)

        for tag in self.tags:
            lines += [
                class_specific_string([tag, f"{precisions[tag]:.4f}", f"{recalls[tag]:.4f}", f"{f_measures[tag]:.4f}"]),
            ]
        lines += [""]

        string = "\n".join(lines)

        if output is None:
            print(string)
        else:
            with open(output, 'w') as f:
                f.write(string)

        if plot:
            # plot the confusion matrix with the use of prettyplot library
            # pretty_plot_confusion_matrix(confusion_mat, cmap='Blues')
            # plot_confusion_matrix_from_data(self.y_true, self.y_pred, columns=self.tags, cmap='Blues')
            cm_display = ConfusionMatrixDisplay(confusion_mat, self.tags).plot(cmap='Blues', values_format='d')
            if output is None:
                plt.show()
            else:
                # save to file
                plt.savefig(output + "_matrix.eps", format='eps')

    def get_measurement(self, fn_measure, averaged=True):
        if averaged:
            return fn_measure(self.y_true, self.y_pred, labels=self.tags, average='weighted', zero_division=0)

        return dict(zip(self.tags, fn_measure(self.y_true, self.y_pred, labels=self.tags, average=None, zero_division=0)))


