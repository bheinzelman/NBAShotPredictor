import sys
sys.path.append('../')
from lib.Table import Table
import math
import json

'''
    This program cleans and discritizes the data
'''

def disc_distance(distance, vals):
    distance = float(distance)
    for i, val in enumerate(vals):
        if distance >= val:
            return i

def get_quartiles(ds, idx, n):
    table = map(lambda r: r[:idx] + [float(r[idx])] + r[idx:], ds.table)
    
    s_table = sorted(table, key=lambda x: x[idx])
    
    quarter_len = len(table)/n
    
    quartiles = [s_table[quarter_len*i: (quarter_len*(i+1))][0][idx] for i in xrange(n)]
    quartiles.reverse()
    return quartiles


if __name__ == '__main__':
    
    ROWS_WE_WANT = [2, 3, 4, 6, 11, 13, 14, 16, 19]
    
    # interesting row IDXs
    distance_idx = 11
    made_idx = 13
    ddist = 16
    margin_idx = 4
    CLOSEST_DEFENDER = 14

    SHOOTER = 19 

    ds = Table(file="shot_logs.csv")

    titles = ds.table[0]

    ds.table = ds.table[1:]
    
    ntiles_shot_distance = get_quartiles(ds, distance_idx, 8)
    print "NTILES shot dist"
    print ntiles_shot_distance

    ntiles_def_distance = get_quartiles(ds, ddist, 5)

    print "NTILES def distance"
    print ntiles_def_distance
    
    ntiles_final_margin = get_quartiles(ds, margin_idx, 10)
    
    print "NTILES final margin"
    print ntiles_final_margin
    
    new_table = Table(table=[])
    
    for row in ds.table:
        # discritize shot distance to a value from 1-10
        val = disc_distance(row[distance_idx], ntiles_shot_distance)
        row[distance_idx] = val
        
        # discritize defender distance to a value from 1-10
        val = disc_distance(row[ddist], ntiles_def_distance)
        row[ddist] = val
        
        val = disc_distance(row[margin_idx], ntiles_final_margin)
        row[margin_idx] = val

        # now get rid of everything we dont want
        row[CLOSEST_DEFENDER] = row[CLOSEST_DEFENDER].replace(',', '')

        new_table.table.append([row[i] for i in ROWS_WE_WANT])
    
    titles = [titles[i] for i in ROWS_WE_WANT]
    
    new_table.table = [titles] + new_table.table
    new_table.file = "shot_log.min.csv"
    new_table.export()
    
    


            


    

        
