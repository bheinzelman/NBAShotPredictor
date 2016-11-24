from Table import Table
import copy
from random import shuffle, randint
import numpy as np
'''
    Class with static methods to return
    stratified sample of a given data set
'''
class StratifiedSample(object):
    '''
        Returns the proper ratio of class values for each
        partition
    '''
    @staticmethod
    def propper_ratios(table, idx, k):
        freq = {}

        # number of instances per partition
        space = table.count_rows() / k

        # count different classes
        for row in table.table:
            if row[idx] in freq:
                freq[row[idx]] += 1
            else:
                freq[row[idx]] = 1

        # change to percentages
        for key in freq.keys():
            freq[key] /= float(table.count_rows())
            freq[key] *= space
            freq[key] = int(freq[key])

        return freq
    '''
        Takes the dataset and splits it into k different partitions each with
        more or less the same distribution of class labels
    '''
    @staticmethod
    def get_partitions(table, idx, k):
        freq = StratifiedSample.propper_ratios(table, idx, k)

        groups = copy.deepcopy(table.group_by(idx, type="map"))

        partitions = []

        # builds each partition with the same ratio of each class
        for i in range(0, k):
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
            partitions[i % k].append(val)
            i += 1

        return partitions

    '''
        Given the partition list, i.e. [d1, d2, ..., dk]
        this function will return a tuple containing di,
        and the union of the other tables, which will be
        the training set.
    '''
    @staticmethod
    def cross_validation(partitions, i):
        cpy = copy.deepcopy(partitions)

        test = partitions[i]
        training = []

        # build up union of partitions for training set
        for j in range(0, len(cpy)):
            if i != j:
                training += cpy[j]

        return (test, Table(table=training))

    @staticmethod
    def stratified_sample(table, k, idx):
        table = copy.deepcopy(table)

        np.random.shuffle(table.table)

        partitions = StratifiedSample.get_partitions(table, idx, k)

        test, training = StratifiedSample.cross_validation(partitions, 0)

        return (Table(table=test), training)


def bootstrap_sample(table):
    table = copy.deepcopy(table)
    np.random.shuffle(table.table)

    idxs = np.random.choice(range(len(table.table)), len(table.table)-1, replace=True)

    sample = [table.table[idx] for idx in idxs]

    return Table(table=sample)
