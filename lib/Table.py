
import csv

'''
    class to handle table operations
'''
class Table(object):
    def __init__(self, **kwargs):
        file = kwargs.get('file', None)
        inTable = kwargs.get('table', None)

        if file is not None:
            self.table = self.get_table(file)
        elif inTable is not None:
            self.table = inTable

        self.file = file

    def get_table(self, file):
        with open(file, 'r') as f:
            reader = csv.reader(f, dialect="excel")

            return [row for row in reader if len(row) > 0]
    '''
        returns a lists of lists grouped together based on
        another attribute
    '''

    def group_by(self, index, **kwargs):
        domain = kwargs.get('domain', [])
        type = kwargs.get('type', 'list')

        groups = {}
        for val in domain:
            groups[val] = []

        for row in self.table:
            if row[index] in groups:
                groups[row[index]].append(row)
            else:
                groups[row[index]] = [row]

        if type == 'map':
            return {key: Table(table=groups[key]) for key in groups.keys()}
        else:
            # fix maybe
            group_list = sorted(
                [groups[key] for key in groups.keys()])
            return [Table(table=group) for group in group_list]

    '''
        returns the number of instances in a table, int
    '''

    def count_rows(self):
        return len(self.table)

    def get_column(self, column):
        return [row[column]
                for row in self.table if row[column] != 'NA']

    # return the min of the table
    def min(self, column):
        return min(self.get_column(column))

    # returns the max of the table
    def max(self, column):
        return max(self.get_column(column))

    # returns the midpoint of the table
    def average(self, column):
        col = self.get_column(column)
        entries = len(col)
        return sum(col) / float(entries)

    # returns the excel formatted string of the table
    def __str__(self):
        s = ""
        for row in self.table:
            for i in range(0, len(row)):
                col = row[i]
                if i != 0:
                    s += ','
                s += str(col)
            s += "\n"
        return s

    '''
        returns all the possible values for a particular
        column as a set
    '''
    def get_vals(self, index):
        vals = set()
        for row in self.table:
            vals.add(row[index])
        return vals
