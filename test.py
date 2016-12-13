from lib.Classifiers import DecisionTreeClassifier, KNNClassifier, BayesianClassifier, RandomForestClassifier
from lib.BBallClassifier import BBallRFClassifier
from lib.dummy import DummyClassifier
from lib.Table import Table
from lib.Test import CrossValidation
from lib.Samples import StratifiedSample
import random
import json

'''
    Program to test how generic classifiers predict 
    if a shot is made or missed..

    NOTE: Knn and Bayes are restricted to a fraction of the
    dataset for time's sake
'''


if __name__ == '__main__':
    CLASS_ID = 5
    ATS1 = [0,1,2,3,4,6,7,8,9]
    ATS = [0,1,2,3,4,7]

    print "Loading table..." 
    t = Table(file="datasets/shot_log.min.csv") 
    t.table = t.table[1:] 
    domain = t.get_domain(ATS1)
    train, test_set = StratifiedSample.stratified_sample(t, 3, CLASS_ID)
    print "Done...\n\n" 


    print 'Building BBall Random Forest Classifier...'
    rf = BBallRFClassifier(CLASS_ID, ATS1, train, 15, 7, 5, domain=domain)
    print 'Done...\n\n'

    print 'Testing BBall Random Forest Classifier...' 
    test = CrossValidation(rf, 4, test_set)
    test.run_test()
    print 'Done...'

    domain = t.get_domain(ATS)

    print 'Testing Random Forest Classifier...' 
    rf = RandomForestClassifier(CLASS_ID, ATS, train, 15, 7, 5, domain=domain)

    test = CrossValidation(rf, 4, test_set)
    test.run_test()
    print 'Done...\n\n'

    print 'Testing decision tree... '
    tree = DecisionTreeClassifier(CLASS_ID, ATS, train, domain=domain)
    test = CrossValidation(tree, 4, test_set)
    test.run_test()
    print 'Done...\n\n'

    print "shuffling rows" 
    random.shuffle(train.table)
    random.shuffle(test_set.table)

    train.table = train.table[:1000]
    test_set.table = test_set.table[:333]
    print "Done...\n\n"

    print "testing... Bayes w/ dataset size 1000"
    bayes = BayesianClassifier(CLASS_ID, ATS1, train)
    test = CrossValidation(bayes, 10, test_set)
    test.run_test()
    print 'Done...\n\n'

    print "testing... KNN w/ dataset size 500"
    CONT_ATS = [2,3,4,7]
    CAT_ATS = [0,1,6,8]

    knnc = KNNClassifier(CLASS_ID, CONT_ATS, CAT_ATS, 5, train) 
    test = CrossValidation(knnc, 10, test_set)
    test.run_test()
    print "Done..."
    
    
