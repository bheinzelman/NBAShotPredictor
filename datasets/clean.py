import sys
sys.path.append('../')
from lib.Table import Table

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
    distance_idx = 11
    made_idx = 13

    ds = Table(file="shot_logs.csv")

    ds.table = ds.table[1:]

    ntiles = get_quartiles(ds, distance_idx, 10)
    
    for row in ds.table:
        val = disc_distance(row[distance_idx], ntiles)
        row[distance_idx] = val

    ds.file = "shot_log.min.csv"
    ds.export()
    


            


    

        
