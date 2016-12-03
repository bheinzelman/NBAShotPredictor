from Classifiers import RandomTreeNode, RandomTreeClassifier, RandomForestClassifier

class BBallTreeNode(RandomTreeNode):
    '''
        For each possible lable, returns the probability
        that that a given instance has that label
    '''
    #OVERRIDE
    def get_class_label(self, table, class_idx):
        freq = {}
        for row in table.table:
            if row[class_idx] in freq:
                freq[row[class_idx]] += 1
            else:
                freq[row[class_idx]] = 1

        fs = {key: freq[key]/float(len(table.table)) for key in freq.keys()}
        return fs

    def make_node(self):
        return BBallTreeNode(self.f, domain=self.domain)

class BBallRTClassifier(RandomTreeClassifier):
    def init_tree(self):
        node = BBallTreeNode(self.f, domain=self.domain).build_children(
            self.table, self.attributes, self.idx)
        return node

    # OVERRIDE
    def set_table(self, new_table):
        node = BBallTreeNode(self.f, domain=self.domain).build_children(
            new_table, self.attributes, self.idx)
        self.tree = node

    def __str__(self):
        return "BBall Random Decision Tree"

class BBallRFClassifier(RandomForestClassifier):
    def __init__(self, idx, attributes, table, n, m, f, **kwargs):
        super(BBallRFClassifier, self).__init__(idx, attributes, table, n, m, f, **kwargs)

        # self.percentages = shot_perc 
        self.SHOOTER = 8
    
    '''
        OVERRIDE
        Given a validation set, and a classifier,
        will test the accuracy of a given classifier
    '''
    def test_accuracy(self, validation_set, tree):
        correct = 0
        for row in validation_set:
            predictions = tree.classify(row)

            predictions = [(key, predictions[key]) for key in predictions.keys()]

            prediction = min(predictions, key=lambda x: x[1])

            if prediction == row[self.idx]:
                correct += 1

        return correct / float(len(validation_set))

    def classify(self, row):
        votes = {}

        for t in self.trees:
            values = t.classify(row)

            for key in values.keys():
                if key not in votes:
                    votes[key] = values[key]
                else:
                    votes[key] += values[key]
 
        # votes = [(key, votes[key]) for key in votes.keys()]
        
        # the_vote = max(votes, key=lambda x: x[1]/float(len(self.trees)))
        
        if 'made' in votes:
            if votes['made']/float(len(self.trees)) >= .53:
                return 'made'
        return 'missed'

        
        return the_vote[0]
    
    def build_single_classifier(self, idx, attributes, boot, f, domain):
        return BBallRTClassifier(idx, attributes, boot, self.f, domain=domain)

    def __str__(self):
        return "BBall Random Forest Classifier"
