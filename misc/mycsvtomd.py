'''
Run from the command line with arguments of the CSV files you wish to convert.
There is no error handling so things will break if you do not give it a well
formatted CSV most likely.

USAGE: python mycsvtomd.py [first_file.csv] [second_file.csv] ...

OUTPUT: first_file.md second_file.md ...
'''
import sys
import csv
##import os
##import re
##import shutil

for arg in sys.argv[1:]:
##    dir_name = arg.split(".")[0] + "_markdown"
##
##    if os.path.exists(dir_name):
##        shutil.rmtree(dir_name)

##    # create a directory to store the results
##    os.mkdir(dir_name)

    if not arg.endswith('.csv'):
        print 'Warning: {} does not end in .csv; skipping'.format(arg)
        continue

    # read in CSV file
    with open(arg, 'rb') as f:
        with open(arg[:-3]+'md', 'wb') as md:
            reader = csv.reader(f)

            # strip off CSV header for names of markdown headers
            header = reader.next()
            md.write('|'.join(header)+'\n')
            for _ in header:
                md.write('|')
            md.write('\n')

            # parse through each record/row and convert to markdown
            for row in reader:
                md.write('|'.join(row)+'\n')
print 'ok'