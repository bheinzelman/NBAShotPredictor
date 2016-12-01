''' 
    File to generate visualizations from the 
    nba shot log
    
    indexes
    LOCATION--------------0
    W---------------------1
    FINAL_MARGIN----------2
    PERIOD----------------3
    SHOT_DIST-------------4
    SHOT_RESULT-----------5
    CLOSEST_DEFENDER------6
    CLOSE_DEF_DIST--------7
    player_name-----------8
'''

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
from lib.Table import Table
import os

def get_shot_dist_ntiles(n, idx):
    SHOT_DIST = 11
    ds = Table(file="datasets/shot_logs.csv")
    ds.table = ds.table[1:]

    table = map(lambda r: r[:idx] + [float(r[idx])] + r[idx:], ds.table)
    
    s_table = sorted(table, key=lambda x: x[idx])
    
    quarter_len = len(table)/n
    
    quartiles = [str(s_table[quarter_len*i: (quarter_len*(i+1))][-1][idx]) for i in xrange(n)]

    # quartiles[0] = '0-' + str(quartiles[0])

    labels = []
    prev = str(0.0)
    for val in quartiles:
        labels.append(str(prev) + '-\n' + str(val))
        prev = val

    return labels 



def made_miss_by_dist(table, idx, other_idx, xlabel, ylabel, name, title):
    RESULT = 5
    groups = table.group_by(idx)
    groups.sort(key=lambda g: g.table[0][idx])

    made = [0 for _ in groups]
    missed = [0 for _ in groups]


    for i, group in enumerate(groups):
        for row in group.table:
            if row[RESULT] == 'made':
                made[i] += 1
            else:
                missed[i] += 1
    
    made.reverse()
    missed.reverse()
    
    pyplot.figure()

    fig, ax = pyplot.subplots()

    r1 = ax.bar(range(1, len(groups) + 1), made, 0.3, color='g')
    r2_v = map(lambda x: x + 0.3, range(1, len(groups) + 1))
    r2 = ax.bar(r2_v, missed, 0.3, color='r')
    
    ax.set_xticks(map(lambda x: x + 0.3, range(1, len(groups) + 1)))
    ax.set_xticklabels(get_shot_dist_ntiles(len(groups), other_idx))

    ax.legend((r1[0], r2[0]), ('Made', 'Missed'), loc=2)

    pyplot.grid(True)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.title(title)
    pyplot.savefig(name)
    pyplot.close()
    

if __name__ == '__main__':
    RESULT = 5

    SHOT_DIST = 4
    SHOT_DIST_ORIG = 11

    DEF_DIST = 7
    DEF_DIST_ORIG = 16

    LOCATION = 0
    LOCATION_ORIG = 2
    
    ds = Table(file="datasets/shot_log.min.csv")

    ds.table = ds.table[1:]

    if not os.path.exists('viz'):
        os.makedirs('viz')

    made_miss_by_dist(ds, SHOT_DIST, SHOT_DIST_ORIG, "Shot Distance (FT)", "Count", "viz/shot_distance.pdf", "Shot Distance")
    made_miss_by_dist(ds, DEF_DIST, DEF_DIST_ORIG, "Defensive Distance (FT)", "Count", "viz/def_distance.pdf", "Defender Distance")

