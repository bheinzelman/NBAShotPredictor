'''
    Program calculates some simple stats on the shot log dataset
'''

import sys
sys.path.append('../')
from lib.Table import Table

'''
    given partitions as a map, returns accuracy per partition
'''
def print_accuracy_by_partitions(partitions, idx):
    percentages = {}

    for key in partitions.keys():
        percentages[key] = shot_accuracy(partitions[key], idx)
    
    lst = [(key, percentages[key]) for key in partitions.keys()]

    lst = sorted(lst, key=lambda x:x[0]) 

    for key, perc in lst:
        perc = int(perc * 100)
        print key + ": " + str(perc) + "%"



def shot_accuracy(table, idx):
    made = 0
    for row in table.table:
        if row[idx] == 'made':
            made += 1.0
    return made/len(table.table)
    
'''
    Returns a list of players with at least 200 shots and their
    shooting percentage
'''
def shot_percentage_by_shooter(table, idx, min_shots):
    # [name]  = [made, total]
    percentages = {}
    shooter_idx = 8

    for row in table.table:
        player = row[shooter_idx]
        
        if player not in percentages:
            percentages[player] = [0, 0]
    
        made = percentages[player][0]
        
        total = percentages[player][1]

        if row[idx] == 'made':
            made += 1.0
        total += 1.0

        percentages[player] = [made, total]

    # made list
    return [(key, percentages[key][0]/percentages[key][1]) for key in percentages.keys() if player[1] > min_shots] 

if __name__ == '__main__':
    shot_log = Table(file="shot_log.min.csv")
    shot_log.table = shot_log.table[1:]
    
    MADE = 5
    SHOT_DIST = 4
    DEF_DIST = 7
    LOCATION = 0
    GAME_OUTCOME = 1
    MARGIN = 2
    SHOT_CLOCK = 9
    PERIOD = 3
    

    print "Total percentage of shots made: " + str(shot_accuracy(shot_log, MADE)) + "%"
    
    shot_percentages = shot_percentage_by_shooter(shot_log, MADE, 500)
    shot_percentages.sort(key=lambda x: x[1], reverse=True)
    
    print "\nTop Shooters"
    for i, player in enumerate(shot_percentages[:10]):
        print str(i + 1) + ': ' + player[0] + ": " + str(int(100*player[1])) + "%"


    print "\nWorst Shooters"
    for i, player in enumerate(shot_percentages[-10:]):
        print str(i + 1) + ': ' + player[0] + ": " + str(int(100*player[1])) + "%"

    
    print "\nAccuracy By Distance"
    print_accuracy_by_partitions(shot_log.group_by(SHOT_DIST, type="map"), MADE)
    
    print "\nAccuracy By Defensive Distance"
    print_accuracy_by_partitions(shot_log.group_by(DEF_DIST, type="map"), MADE)

    print "\nAccuracy By Game Outcome (Win/Lose)"
    print_accuracy_by_partitions(shot_log.group_by(GAME_OUTCOME, type="map"), MADE)

    print "\nAccuracy By Location (Home/Away)"
    print_accuracy_by_partitions(shot_log.group_by(LOCATION, type="map"), MADE)
    
    print "\nAccuracy By Score Offset"
    print_accuracy_by_partitions(shot_log.group_by(MARGIN, type="map"), MADE)
    
    print "\nAccuracy By Shot Clock"
    print_accuracy_by_partitions(shot_log.group_by(SHOT_CLOCK, type="map"), MADE)

    print "\nAccuracy By Period"
    print_accuracy_by_partitions(shot_log.group_by(PERIOD, type="map"), MADE)
