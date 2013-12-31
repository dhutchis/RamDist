from __future__ import division
import igraph
import csv
import itertools
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# terminology:  good graph has a K3 or an I4
#               bad graph has neither a K3 nor an I4

# Goal: for R(3,4)=9, determine the distribution of the proportion of good graphs of size n
#       as n increases from 1 to 9
def __main__():
    dic = dict()
    for n in range(1,7):
        dic[n] = goodfraction_systematic(n,3,3)
    #dic = create_dist(3,3)
    print dic
    writer = csv.writer(open('dict.csv', 'wb'))
    for key, value in dic.items():
        writer.writerow([key, value])

# Given q1, q2, create dict {1:0, 2:0, 3:0.05}
# alternative? [(1,0),(2,0),(3,0.05),...]
def create_dist(q1,q2):
    dic = dict()
    for n in range(1,7):
        dic[n] = goodfraction_systematic(n,q1,q2)
        if dic[n] == 1 and (n == 1 or dic[n-1] == 1):
            break
    return dic
    for n in itertools.count(7):
        dic[n] = goodfraction_sample(n,q1,q2,10000) # fix sample size 10000
        if dic[n] == 1 and dic[n-1] == 1:
            break
    return dic

# Given n (<10), go through every graph of size n and count how many have are good and return the fraction
# optimization 1 if q1==q2: only go through half the graphs, since the complement is the same
def goodfraction_systematic(n, q1, q2):
    if n < q1 and n < q2:
        return 0.0
    if n >= 10:
        print 'warning - this may take a while for n = ', n
    #g = igraph.Graph(n) # empty graph
    good = 0
    bad = 0
    for g in generate_graphs(n, q1==q2):
        # check graph - is it good?
        #  find max clique # mc
        mc = g.omega()
        if mc >= q1: # then good graph
            good += 2 if q1==q2 else 1
            logger.debug('n={}, {}, omega={}, good'.format(n,g,mc))
        else:
            #  find max independent set # mi
            mi = g.alpha()
            if mi >= q2: # then good graph
                good += 2 if q1==q2 else 1
                logger.debug('n={}, {}, omega={}, alpha={} good'.format(n,g,mc,mi))
            else: #  bad graph!
                bad += 2 if q1==q2 else 1
                logger.debug('n={}, {}, omega={}, alpha={} BAD'.format(n,g,mc,mi))
    assert good + bad == 2 ** (n*(n-1)/2)
    return good / (good + bad)

# edge 5-4 5-3 5-2 5-1

# if do_half is True, only run through the first half of the graphs.
#   The second half are all complements of the first.
def generate_graphs(n, do_half):
    g = igraph.Graph(n)
    if do_half:
        for gg in generate_graphs_help(g,n-1,n-3):
            yield gg
    else:
        for gg in generate_graphs_help(g,n-1,n-2):
            yield gg
    return

def generate_graphs_help(g,i,j):
    if i<=-1:
        yield g
    elif j<=-1:
        for gg in generate_graphs_help(g,i-1,i-2):
            yield gg
    else:
        for gg in generate_graphs_help(g,i,j-1):
            yield gg
        g.add_edge(i,j)
        for gg in generate_graphs_help(g,i,j-1):
            yield gg
        g.delete_edges((i,j))
    return

# Given n and sample size m, sample m graphs of size n, count how many are good and return the fraction
def goodfraction_sample(n,q1,q2,m):
    print 'todo'


__main__()