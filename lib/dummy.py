from Classifiers import Classifier

class DummyClassifier(Classifier):
    def __init__(self, idx, attributes, table):
        super(DummyClassifier, self).__init__(idx, attributes, table)

    def classify(self, row):
        return "missed"

    def __str__(self):
        return "Dummy Classifier"

