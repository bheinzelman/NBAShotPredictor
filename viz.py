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
# matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
from lib.Table import Table
import os

def remove_shotclock_nas(table):
    shotclock = 8
    gameclock = 7

    for row in table.table:
        if len(row[shotclock]) == 0:
            new_time = row[gameclock].replace(':', '.')

            row[shotclock] = new_time

def get_shot_dist_ntiles(n, idx):
    ds = Table(file="datasets/shot_logs.csv")
    ds.table = ds.table[1:]

    remove_shotclock_nas(ds)

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



def bar_graph_continuous(table, idx, other_idx, xlabel, ylabel, name, title):
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

def bar_graph_categorical(table, idx, xlabel, ylabel, name, title, **kwargs):
    RESULT = 5
    
    location = kwargs.get('loc', 2)

    groups = table.group_by(idx, type="map") 

    made = [0 for key in groups.keys()]
    missed = [0 for key in groups.keys()]

    key_map = {key: i for i, key in enumerate(groups.keys())}

    for key in groups.keys():
        for row in groups[key].table:
            if row[RESULT] == 'made':
                made[key_map[key]] += 1
            else:
                missed[key_map[key]] += 1

    group_count = len(groups.keys())

    pyplot.figure()

    fig, ax = pyplot.subplots()

    r1 = ax.bar(range(1, group_count + 1), made, 0.3, color='g')
    r2_v = map(lambda x: x + 0.3, range(1, group_count + 1))
    r2 = ax.bar(r2_v, missed, 0.3, color='r')
    
    ax.set_xticks(map(lambda x: x + 0.3, range(1, group_count + 1)))

    ax.legend((r1[0], r2[0]), ('Made', 'Missed'), loc=location)
    ax.set_xticklabels(key_map.keys())

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

    GAME_OUTCOME = 1

    MARGIN = 2
    MARGIN_ORIG = 4

    SHOT_CLOCK = 9
    SHOT_CLOCK_ORIG = 8

    PERIOD = 3
 
    ds = Table(file="datasets/shot_log.min.csv")

    ds.table = ds.table[1:]

    if not os.path.exists('viz'):
        os.makedirs('viz')

    bar_graph_continuous(ds, SHOT_DIST, SHOT_DIST_ORIG, "Shot Distance (FT)", "Count", "viz/shot_distance.png", "Shot Distance")
    bar_graph_continuous(ds, DEF_DIST, DEF_DIST_ORIG, "Defensive Distance (FT)", "Count", "viz/def_distance.png", "Defender Distance")
    bar_graph_continuous(ds, MARGIN, MARGIN_ORIG, "Final Margin (PTS)", "Count", "viz/margin.png", "Shots by Final Margin")
    bar_graph_continuous(ds, SHOT_CLOCK, SHOT_CLOCK_ORIG, "Shot Clock (seconds)", "Count", "viz/shot_clock.png", "Shots by Shotclock")

    bar_graph_categorical(ds, LOCATION, "Location H/A", "Count", "viz/home_away.png", "Shots by Location")
    bar_graph_categorical(ds, GAME_OUTCOME, "Game Outcome W/L", "Count", "viz/win_lose.png", "Shots by Game Outcome")
    bar_graph_categorical(ds, PERIOD, "Period", "Count", "viz/period.png", "Shots by Period", loc=1)

