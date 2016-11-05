from random import randint
import copy
from tabulate import tabulate
from Table import Table

'''
    Author Bert Heinzelman, Brad Carrion
    Date: 10/14/16
    Description: file to hold all test classes
'''


'''
    Base class for Test hierarchy.

    Given a classifier and a table, will run
    tests on the dataset
'''


class Test(object):

    def __init__(self, classifier, table):
        self.table = table
        self.classifier = classifier
        self.idx = self.classifier.idx
        self.confusion = self.generate_confusion_matrix()

    '''
        Executes the tests
    '''

    def run_test(self):
        raise Exception("Cannot call method from base class")

    '''
        Fills out the confusion matrix
    '''

    def generate_confusion_matrix(self):
        distinct_vals = self.table.get_vals(self.idx)
        x_len = 3 + len(distinct_vals)

        matrix = []
        for i in distinct_vals:
            new_row = [0 for j in range(x_len)]
            new_row[0] = i 
            matrix.append(new_row)

        return matrix

    '''
        prints the confusion matrix
    '''

    def print_matrix(self):
        distinct_vals = list(self.table.get_vals(self.idx))
        header = [None for i in range(len(distinct_vals) + 3)]
        header[0] = "Value"
        for i in range(1, len(distinct_vals) + 1):
            val = distinct_vals[i - 1]
            header[i] = val
        header[-1] = "Recognition (%)"
        header[-2] = "Total"

        print "\n"
        print tabulate(self.confusion, header, tablefmt="rst")
        print "\n"

    '''
        Calculates the multi class accuracy with the
        confusion matrix
    '''

    def accuracy(self):
        pure_matrix = []
        total = sum([self.confusion[i][-2]
                     for i in range(len(self.confusion))])

        accuracy = 0

        for row in self.confusion:
            pure_matrix.append(row[1:])

        for i in range(len(pure_matrix)):
            tp = pure_matrix[i][i]

            tn = 0
            for j in range(len(pure_matrix)):
                for k in range(len(pure_matrix)):
                    if j == i or k == i:
                        continue
                    else:
                        tn += pure_matrix[j][k]
            try:
                accuracy += (float(tp) + tn) / total
            except:
                pass

        distinct_vals = list(self.table.get_vals(self.idx))
        return accuracy / float(len(distinct_vals))

    '''
        Calculates error with the confusion matrix
    '''

    def error(self):
        return 1 - self.accuracy()

    '''
        Generates results in the confusion matrix,
        e.g. total, and recognition
    '''

    def generate_results(self):
        distinct_vals = list(self.table.get_vals(self.idx))

        # compute total column and recognition
        for i in range(0, len(distinct_vals)):
            self.confusion[i][-2] = sum(self.confusion[i][1:-2])

            try:
                self.confusion[i][-1] = self.confusion[i][i +
                                                          1] / float(self.confusion[i][-2])
            except:
                self.confusion[i][-1] = "NA"


'''
    Performs a random test, takes 5 rows and trys to classify them
'''


class RandomTest(Test):

    def run_test(self):
        runs = 5
        for i in range(0, runs):
            rindex = randint(0, len(self.table.table) - 1)
            instance = self.table.table[rindex]
            predicted_val = self.classifier.classify(instance)
            s_instance = (str(instance)[1:-1]).replace('\'', '')
            print 'instance: ' + s_instance
            print 'class: ' + str(int(predicted_val)) + ', actual: ' + str(instance[self.classifier.idx])

'''
    Perfoms a random subsampling test run.
    give classifier, rounds(times to run the tests), split (fraction of test
    compared to training), and a table
'''


class RandomSubsample(Test):

    def __init__(self, classifier, rounds, split, table):
        super(RandomSubsample, self).__init__(classifier, table)
        self.rounds = rounds
        self.split = split

    def run_test(self):
        for i in range(self.rounds):
            test, training = self.random_subsample()
            self.classifier.set_table(training)
            for row in test:

                prediction = self.classifier.classify(row)

                self.confusion[row[self.idx] - 1][prediction] += 1

        self.generate_results()
        # self.print_matrix()
        print "\nRandom Subsample (k=10, 2:1 Train/Test)"
        print "\t " + str(self.classifier) + ": accuracy: " + str(self.accuracy()) + ", error rate: " + str(self.error())

    '''
        Returns a Random subsampling
    '''

    def random_subsample(self):
        copy_table = copy.deepcopy(self.table)

        test_set = []

        # adds rows to test set, deletes from table
        while len(test_set) < len(self.table.table) * self.split:
            idx = randint(0, len(copy_table.table) - 1)
            test_set.append(copy_table.table[idx])

            del copy_table.table[idx]

        return (test_set, copy_table)

'''
    Perfoms Cross validation test runs.
    give k (number of partitions), the classifier,
    and the table
'''


class CrossValidation(Test):

    def __init__(self, classifier, k, table):
        super(CrossValidation, self).__init__(classifier, table)
        self.classifier = classifier
        self.k = k

    '''
        Builds up a frequency table.
        Supplies how many of each class label should be in each partition
    '''

    def propper_ratios(self):
        freq = {}

        # number of instances per partition
        space = self.table.count_rows() / self.k

        # count different classes
        for row in self.table.table:
            if row[self.idx] in freq:
                freq[row[self.idx]] += 1
            else:
                freq[row[self.idx]] = 1

        # change to percentages
        for key in freq.keys():
            freq[key] /= float(self.table.count_rows())
            freq[key] *= space
            freq[key] = int(freq[key])

        return freq

    '''
        Takes the dataset and splits it into k different partitions each with
        more or less the same distribution of class labels
    '''

    def get_partitions(self):
        freq = self.propper_ratios()

        groups = copy.deepcopy(self.table.group_by(self.idx, type="map"))

        partitions = []

        # builds each partition with the same ratio of each class
        for i in range(0, self.k):
            partitions.append([])
            # grabs the correct distribution
            for key in freq.keys():
                values_to_get = freq[key]
                # puts the correct distribution in each partition
                for j in range(0, values_to_get):
                    partitions[-1].append(groups[key].table[0])
                    del groups[key].table[0]

        # sprinkle the rest in some partition
        flat = [val for key in groups.keys() for val in groups[key].table]  # flatten list

        i = 0
        for val in flat:
            partitions[i % 10].append(val)
            i += 1

        return partitions

    '''
        Given the partition list, i.e. [d1, d2, ..., dk]
        this function will return a tuple containing di,
        and the union of the other tables, which will be
        the training set.
    '''

    def cross_validation(self, partitions, i):
        cpy = copy.deepcopy(partitions)

        test = partitions[i]
        training = []

        # build up union of partitions for training set
        for j in range(0, len(cpy)):
            if i != j:
                training += cpy[j]

        return (test, Table(table=training))

    def run_test(self):
        partitions = self.get_partitions()

        distinct = list(set(self.table.get_column(self.idx)))

        #create a lookup for values
        lookup = {val: i for i, val in enumerate(distinct)}

        for i in range(self.k):
            test, training = self.cross_validation(partitions, i)
            self.classifier.set_table(training)
            for row in test:
                prediction = self.classifier.classify(row)

                self.confusion[lookup[row[self.idx]]][lookup[prediction]+1] += 1

        self.generate_results()
        print "Stratified 10-fold Cross Validation"
        print "\t " + str(self.classifier) + ": accuracy: " + str(self.accuracy()) + ", error rate: " + str(self.error())
        self.print_matrix()
