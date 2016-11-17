from lib.Classifiers import DecisionTreeClassifier, KNNClassifier, BayesianClassifier
from lib.Table import Table
from lib.Test import CrossValidation
import random

'''
    Program to test how generic classifiers predict 
    if a shot is made or missed..

    NOTE: Knn and Bayes are restricted to a fraction of the
    dataset for time's sake
'''

if __name__ == '__main__':
    CLASS_ID = 5
    ATS = [0,1,2,3,4,6,7,8]
    t = Table(file="datasets/shot_log.min.csv") 

    print 'Testing decision tree... '
    tree = DecisionTreeClassifier(CLASS_ID, ATS, t)
    test = CrossValidation(tree, 4, t)
    test.run_test()
    print 'Done...'

    print "shuffling" 
    random.shuffle(t.table)
    t.table = t.table[:2000]
    print "Done..."

    print "testing... Bayes w/ dataset size 2000"
    bayes = BayesianClassifier(CLASS_ID, ATS, t)
    test = CrossValidation(bayes, 4, t)
    test.run_test()
    print 'Done...'

    print "testing... KNN w/ dataset size 2000"
    knnc = KNNClassifier(CLASS_ID, [], ATS, 5, t)
    test = CrossValidation(knnc, 4, t)
    test.run_test()
    print "Done..."
    
