from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


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

    def get_classification_report(self, output=None):
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
            f"{confusion_mat}",
            "",
            f"Classification accuracy: {ca}",
            f"Precision: {precision}",
            f"Recall: {recall}",
            f"F1 score: {f_measure}",
            "",
        ]

        precisions = self.get_precision(averaged=False)
        recalls = self.get_recall(averaged=False)
        f_measures = self.get_f1_measure(averaged=False)

        for tag in self.tags:
            lines += [
                f"Class: {tag}",
                f"Precision: {precisions[tag]}",
                f"Recall: {recalls[tag]}",
                f"F1 score: {f_measures[tag]}",
                "",
            ]

        string = "\n".join(lines)

        if output is None:
            print(string)
        else:
            with open(output) as f:
                f.write(string)

    def get_measurement(self, fn_measure, averaged=True):
        if averaged:
            return fn_measure(self.y_true, self.y_pred, labels=self.tags, average='weighted')

        return dict(zip(self.tags, fn_measure(self.y_true, self.y_pred, labels=self.tags, average=None)))


