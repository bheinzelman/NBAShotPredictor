import numpy as np
import math
import heapq_max # must pip install heapq_max

'''
    Author Bert Heinzelman, Brad Carrion
    Date: 10/14/16
    Description: file to hold all classifiers
'''


'''
    Base class for classifier object.
    given a table, idx, and set of attributes,
    this class will classify a given rows label
    at index idx
'''
class Classifier(object):
    def __init__(self, idx, attributes, table):
        self.idx = idx
        self.attributes = attributes
        self.table = table

    '''
        params:
            row: the row to classify
            attributes: the rows to include in the classification
            idx: the attribute to predict
    '''
    def classify(self, row):
        raise Exception("Method not implemented in base class")

    def set_table(self, new_table):
        self.table = new_table


'''
    Uses bayes method as a classification technique
'''
class BayesianClassifier(Classifier):

    def __init__(self, idx, attributes, table):
        super(BayesianClassifier, self).__init__(idx, attributes, table)

    def classify(self, row):

        probabilities = []

        for cls in self.table.get_vals(self.idx):
            probabilities.append((self.prob(row, cls), cls))

        probabilities.sort(key=lambda x: x[0])
        return probabilities[-1][1]

    '''
        P(CLASS, ROW)
    '''
    def prob(self, row, cls):
        return self.prob_x_h(row, cls) * self.prob_h(cls)

    '''
        P(ROW, CLASS) with independence assumption
    '''
    def prob_x_h(self, row, cls):
        groups = self.table.group_by(self.idx, type="map")

        prob = 1
        for i in self.attributes:
            vi = row[i]

            rows_with_cls = groups[cls].table

            matches = len(filter(lambda ins: ins[i] == vi, rows_with_cls))

            prob *= (float(matches)/len(rows_with_cls))

        return prob

    '''
        P(CLASS)
    '''
    def prob_h(self, cls):
        groups = self.table.group_by(self.idx, type="map")

        rows_with_cls = groups[cls].table
        return len(rows_with_cls)/float(len(self.table.table))

    def __str__(self):
        return "Naive Bayes"

'''
    Uses Bayisian classification, but uses gaussian function
    for continuous attributes
'''
class DiverseBayesianClassifier(BayesianClassifier):
    def __init__(self, idx, cont_attrs, cat_attrs, table):
        super(DiverseBayesianClassifier, self).__init__(idx, cat_attrs, table)
        self.cont_attrs = cont_attrs

    def prob_x_h(self, row, cls):
        prob = super(DiverseBayesianClassifier, self).prob_x_h(row, cls)

        for i in self.cont_attrs:
            stdev = np.std(self.table.get_column(i))
            mean = self.table.average(i)

            prob *= self.gaussian(float(row[i]), mean, stdev)

        return prob

    def gaussian(self, value, mean, stdev):
        first, second = 0, 0

        if stdev > 0:
            first = 1/(math.sqrt(2*math.pi)*stdev)
            second = math.e**((-(value-mean)**2)/(2*(stdev**2)))

        return first*second


'''
    Classification using KNN
'''
class KNNClassifier(Classifier):
    def __init__(self, idx, attributes, cat_attributes, k, table):
        super(KNNClassifier, self).__init__(idx, attributes, table)
        self.k = k
        self.cat_attributes = cat_attributes

    def classify(self, row):
        closest = self.knn(row, self.k, self.attributes)

        # majority voting
        votes = {}
        for idx in closest:
            row = self.table.table[idx[0]]

            if row[self.idx] in votes:
                votes[row[self.idx]] += 1
            else:
                votes[row[self.idx]] = 1

        prediction = max([(key, votes[key]) for key in votes.keys()], key=lambda x: x[1])

        return prediction[0]

    def knn(self, predicterRow, k, attributes):
        attribute_count = len(self.table.table[0])

        min_max = {i: (self.table.min(i), self.table.max(i)) for i in attributes}

        distances = []

        for j in range(0, len(self.table.table)):
            row = self.table.table[j]
            distance = 0
            for i in attributes:
                training_val = row[i]
                p_val = predicterRow[i]

                if i in self.cat_attributes:
                    if training_val != p_val:
                        distance += 1
                else:
                    normalized_t = self.normalize(training_val, min_max[i][0], min_max[i][1])

                    normalized_p = self.normalize(p_val, min_max[i][0], min_max[i][1])

                    distance += (normalized_t - normalized_p)**2
            
            distance = distance ** .5
            # distances.append((j, distance**(.5)))
            if len(distances) < k:
                heapq_max.heappush_max(distances, (distance, j))
            elif distance < distances[0]:
                heapq_max.heappop_max(distances)
                heapq_max.heappush_max(distances, (distance, j))

        nearest = [heapq_max.heappop_max(distances) for _ in distances]

        # for i in range(0, k):
            # min_dist = min(distances, key=lambda x: x[1])
            # nearest.append(min_dist)
            # distances.remove(min_dist)

        # exclude first elem because it is equal to the row you are trying to
        # find the neighbors of.
        return [(dist[1], dist[0]) for dist in nearest]

    '''
        Normalize a value with respec to its min and max
    '''
    def normalize(self, value, min_val, max_val):
        return (float(value) - float(min_val)) / (float(max_val) - float(min_val))

    def __str__(self):
        return "K Nearest Neighbors"


'''
    Classifier based off of decision tree
'''
class DecisionTreeClassifier(Classifier):
    def __init__(self, idx, attributes, table):
        super(DecisionTreeClassifier, self).__init__(idx, attributes, table)
        self.tree = TreeNode().build_children(table, attributes, idx)
        self.idx = idx
        self.attributes = attributes

    def classify(self, row):
        return self.__classify__(row, self.tree)

    def __classify__(self, row, tree):
        if tree.class_label is not None:
            return tree.class_label
        else:
            value = row[tree.attribute_idx]

            return self.__classify__(row, tree.children[value])

    # regen the table
    def set_table(self, new_table):
        self.tree = TreeNode().build_children(new_table, self.attributes, self.idx)

    def __str__(self):
        return "Decision Tree Classifier"

    def get_rules(self):
        return self.__get_rules__(self.tree, "IF")

    def __get_rules__(self, node, work):
        if node.class_label is not None:
            #we hit a leaf
            work += " THEN class = " + str(node.class_label) + "\n"
            return [work]
        else:
            #just another regular node
            prefix = " AND "
            if work == "IF":
                prefix = " "
            out = []
            for key in node.children.keys():
                child = node.children[key]
                tmp = work

                tmp += prefix + str(node.attribute_idx) + " = " + key
                out += self.__get_rules__(child, tmp)
            return out


'''
Class Variables
    children: this is an array that holds all children nodes to self
'''
class TreeNode(object):

    EMPTY_PARTITION = 0
    NO_ATTRIBUTES = 1
    SUCCESS = 2
    MAKE_SELF_LEAF = 3

    def __init__(self):
        self.children = {}
        self.attribute_idx = None
        self.class_label = None

    def build_children(self, table, attributes, class_idx):
        domain = {}
        for at in attributes:
            domain[at] = list(set(table.get_column(at)))

        return self.__build_children__(table, attributes, class_idx, domain=domain)

    def __build_children__(self, table, attributes, class_idx, domain):
        # check if there are no more attributes
        if len(attributes) == 0:
            return self.NO_ATTRIBUTES

        # check if there are no more instances
        # if len(table.table) == 0:
            # return self.EMPTY_PARTITION

        # check if partitions are all the same classes
        distinct_vals = list(set(table.get_column(class_idx)))
        if len(distinct_vals) == 1:
            self.class_label = distinct_vals[0]
            return self

        enews = [(i, self.calculate_enew(i, class_idx, table)) for i in attributes]

        choice_idx = min(enews, key=lambda x: x[1])

        self.attribute_idx = choice_idx[0]

        child_attributes = filter(lambda x: x != choice_idx[0], attributes)

        partitions = table.group_by(self.attribute_idx, domain=domain[self.attribute_idx])

        for p in partitions:
            if len(p.table) == 0:
                self.children = {}
                self.class_label = self.get_class_label(table, class_idx)
                return self


        for partition in partitions:
            if len(partition.table) > 0:
                at = partition.table[0][self.attribute_idx]
                child = TreeNode()
                result = child.__build_children__(partition, child_attributes, class_idx, domain)
                if result == self.NO_ATTRIBUTES:
                    # find the distribution of this partition, and make this node
                    # a leaf
                    leaf = TreeNode()
                    leaf.class_label = self.get_class_label(partition, class_idx)
                    self.children[at] = leaf
                elif result == self.EMPTY_PARTITION:
                    # we ran out of rows in a partition.. so we need to go up a level and
                    # make it a leaf, so we pass message up to make parent node
                    # a leaf
                    self.children = {}
                    return self.MAKE_SELF_LEAF
                elif result == self.MAKE_SELF_LEAF:
                    self.children = {}
                    self.class_label = self.get_class_label(table, class_idx)
                    return self
                else:
                    self.children[at] =  child

        return self


    '''
        Runs through a table and returns the most occuring class label
    '''
    def get_class_label(self, table, class_idx):
        freq = {}
        for row in table.table:
            if row[class_idx] in freq:
                freq[row[class_idx]] += 1
            else:
                freq[row[class_idx]] = 1
        lst = [(key, freq[key]) for key in freq.keys()]
        return max(lst, key=lambda x: x[1])[0]


    def calculate_enew(self, index, class_idx, table):
        d = table.count_rows()
        freq = self.att_freq(index, class_idx, table)
        e_new = 0

        for attr_val in freq:
            d_j = float(freq[attr_val][1])
            probs = [(t/d_j) for (_, t) in freq[attr_val][0].items()]

            try:
                entropy = -sum([p * math.log(p,2) for p in probs])
            except:
                entropy = 0

            e_new += (d_j/d) * entropy

        return e_new

    def att_freq(self, index, class_idx, table):
        att_vals = list(set(table.get_column(index)))
        class_vals = list(set(table.get_column(class_idx)))

        result = {v: [{c: 0 for c in class_vals}, 0] for v in att_vals}

        for row in table.table:
            label = row[class_idx]
            att_val = row[index]

            result[att_val][0][label] += 1
            result[att_val][1] += 1

        return result
