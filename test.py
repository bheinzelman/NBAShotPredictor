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

def test_classifiers(CLASS_ID, ATS, train, test_set, domain):
    print 'Testing Dummy Classifier (always guess missed)...'
    dummy = DummyClassifier(CLASS_ID, ATS, train)
    test = CrossValidation(dummy, 4, test_set)
    test.run_test()
    print 'Done...\n\n'

    print 'Testing decision tree... '
    tree = DecisionTreeClassifier(CLASS_ID, ATS, train, domain=domain)
    test = CrossValidation(tree, 4, test_set)
    test.run_test()
    print 'Done...\n\n'

    print 'Testing Random Forest Classifier...' 
    print 'Building Random Forest...'
    rf = RandomForestClassifier(CLASS_ID, ATS, train, 15, 7, 5, domain=domain)
    print 'Done...\n\n'

    test = CrossValidation(rf, 4, test_set)
    test.run_test()

    print "shuffling rows" 
    random.shuffle(train.table)
    random.shuffle(test_set.table)

    train.table = train.table[:1000]
    test_set.table = test_set.table[:333]
    print "Done...\n\n"

    print "testing... Bayes w/ dataset size 500"
    bayes = BayesianClassifier(CLASS_ID, ATS, train)
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
    

if __name__ == '__main__':
    CLASS_ID = 5
    ATS1 = [0,1,2,3,4,6,7]
    ATS = [0,1,2,3,4,7]

    print "Loading table..." 
    t = Table(file="datasets/shot_log.min.csv") 
    t.table = t.table[1:]

    #percentages = json.loads(open('datasets/shot_perc.json', 'r').read())

    # shorten up dataset for testing
    #t.table = t.table[:2000]
    
    domain = t.get_domain(ATS1)

    train, test_set = StratifiedSample.stratified_sample(t, 3, CLASS_ID)

    print "Done...\n\n" 

    print 'Building BBall Random Forest Classifier...'
    rf = BBallRFClassifier(CLASS_ID, ATS, train, 15, 7, 5, domain=domain)
    print 'Done...\n\n'

    print 'Testing BBall Random Forest Classifier...' 
    test = CrossValidation(rf, 4, test_set)
    test.run_test()
    print 'Done...'

    domain = t.get_domain(ATS)

    print 'Testing Random Forest Classifier...' 
    print 'Building Random Forest...'
    rf = RandomForestClassifier(CLASS_ID, ATS, train, 15, 7, 5, domain=domain)
    print 'Done...\n\n'

    test = CrossValidation(rf, 4, test_set)
    test.run_test()

    print 'Testing decision tree... '
    tree = DecisionTreeClassifier(CLASS_ID, ATS, train, domain=domain)
    test = CrossValidation(tree, 4, test_set)
    test.run_test()
    print 'Done...\n\n'
    
    # Test all the other ones, knn, random forest, etc.
    # test_classifiers(CLASS_ID, ATS, train, test_set, domain)
