from lib.Classifiers import DecisionTreeClassifier, KNNClassifier, BayesianClassifier, RandomForestClassifier
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
    # ATS = [0,1,2,3,4,6,7,8]
    ATS = [0,1,2,3,4,7]

    print "Loading table..." 
    t = Table(file="datasets/shot_log.min.csv") 
    print "Done...\n\n" 

    print 'Testing decision tree... '
    tree = DecisionTreeClassifier(CLASS_ID, ATS, t)
    test = CrossValidation(tree, 4, t)
    test.run_test()
    print 'Done...'

    print 'Testing Random Forest Classifier...' 
    print 'Building Random Forest...'
    rf = RandomForestClassifier(CLASS_ID, ATS, t, 15, 7, 5, domain=t.get_domain(ATS))
    print 'Done...'

    test = CrossValidation(rf, 4, t)
    test.run_test()

    print "shuffling" 
    random.shuffle(t.table)
    t.table = t.table[:500]
    print "Done..."

    print "testing... Bayes w/ dataset size 500"
    bayes = BayesianClassifier(CLASS_ID, ATS, t)
    test = CrossValidation(bayes, 10, t)
    test.run_test()
    print 'Done...'

    print "testing... KNN w/ dataset size 500"
    CONT_ATS = [2,3,4,7]
    CAT_ATS = [0,1,6,8]

    knnc = KNNClassifier(CLASS_ID, CONT_ATS, CAT_ATS, 5, t) 
    test = CrossValidation(knnc, 10, t)
    test.run_test()
    print "Done..."
    
