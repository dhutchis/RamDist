from __future__ import division
import igraph
import csv
import itertools
import logging
import math
import numpy
import scipy
import scipy.stats
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# terminology:  good graph has a K3 or an I4
#               bad graph has neither a K3 nor an I4

# Goal: for R(3,4)=9, determine the distribution of the proportion of good graphs of size n
#       as n increases from 1 to 9
def __main__():
#    q1, q2 = 4,4
    m = 50000
    # dic is of the form n -> (p, m, lb, ub)
    for q1 in range(3,7):
        for q2 in range(3,q1+1):
            dic = dict()
            #dic = create_dist(q1,q2,m)
            print q1,q2,dic
            #write_results_to_file(dic,q1,q2,m)
            dic = read_results_from_file(q1,q2,m)
            do_plot(dic,q1,q2,m)

def write_results_to_file(dic,q1,q2,m):
    'Write dictionary of results to csv file'
    writer = csv.writer(open('results/dic_R{}{}_m{}.csv'.format(q1,q2,m), 'wb'))
    for key, value in dic.items():
        p, m, lb, ub = value
        writer.writerow([key, p, int(m), lb, ub])

def read_results_from_file(q1,q2,m):
    'Read csv file and return the dictionary of values'
    reader = csv.reader(open('results/dic_R{}{}_m{}.csv'.format(q1,q2,m), 'rb'))
    dic = dict()
    for row in reader:
        dic[int(row[0],10)] = float(row[1]),int(row[2],10),float(row[3]),float(row[4]) #tuple(map(float,row[1:5]))
        #print dic[int(row[0])]
    return dic

def do_plot(dic,q1,q2,m):
    x_n, y_p, y_lb, y_ub = [], [], [], []
    for key, val in dic.items():
        x_n.append(key)
        y_p.append(val[0])
        y_lb.append(val[0]-val[2]) # want errors from center
        y_ub.append(val[3]-val[0])
    # get lb, ub
    lb = get_lb(q1,q2)
    ub = get_ub(q1,q2)
    plt.hold(False)
    #plt.plot(x_n, y_lb, 'r')
    #plt.plot(x_n, y_ub, 'r')
    plt.plot(x_n, y_p, linestyle="dashed", marker="o") #linewidth=2,
    plt.hold(True)
    plt.errorbar(x_n,y_p, yerr=[y_lb,y_ub], linestyle="None", marker="None")
    if lb != None:
        plt.axvline(x=lb, ymin=0, ymax=1, color='g', linestyle='dashed')
    if ub != None:
        plt.axvline(x=ub, ymin=0, ymax=1, color='m', linestyle='dashed')
        plt.axvline(x=ub+ub/25,ymin=0,ymax=0,linestyle="None", marker="None")
    plt.xlabel('n', size=18)
    plt.ylabel('Pr( good graph )', size=18)
    plt.title('q1={}, q2={}'.format(q1,q2), size=20)
    #plt.show()
    plt.savefig('results/pic_R{}{}_m{}.png'.format(q1,q2,m))

# known lower bounds
arr_lb = [
[1],
[1,2],
[1,3,6],
[1,4,9,18],
[1,5,14,25,43],
[1,6,18,36,58,102],
[1,7,23,49,80,113,205],
[1,8,28,58,101,132,217,282],
[1,9,36,73,126,169,241,317,565],
[1,10,40,92,144,179,289,331,581,798]
]
arr_ub = [
[1],
[1,2],
[1,3,6],
[1,4,9,18],
[1,5,14,25,49],
[1,6,18,41,87,165],
[1,7,23,61,143,298,540],
[1,8,28,84,216,495,1031,1870],
[1,9,36,115,316,780,1713,3583,6588],
[1,10,42,149,442,1171,2826,6090,12677,23556]
]
def get_lb(q1,q2):
    if q1<q2:
        q1, q2 = q2, q1
    q1 -= 1 # adjust index to 0-based
    q2 -= 1
    if q2 < 0 or q1 >= len(arr_lb):
        return None
    return arr_lb[q1][q2]
def get_ub(q1,q2):
    if q1<q2:
        q1, q2 = q2, q1
    q1 -= 1 # adjust index to 0-based
    q2 -= 1
    if q2 < 0 or q1 >= len(arr_lb):
        return None
    return arr_ub[q1][q2]


# 0.5 is conservative; changing makes a smaller n
##def choose_sample_size(confidence, interval_width, p=0.5, df=1000):
##    '''Choose a sample size given confidence, width, proportion p defaulting to conservative 0.5,
##    degrees of freedom defaulting to 1000.
##    Min. sample size 1000.'''
##    tstar = scipy.stats.t.ppf(1- (1-confidence)/2, 1000 if df <= 1000 else df)
##    s = math.sqrt(p*(1-p))
##    n = math.ceil(tstar * s / interval_width) ** 2
##    return n if n > 1000 else 1000

# Given q1, q2, create dict
def create_dist(q1,q2,m):
    dic = dict()
    for n in range(1,7):
        dic[n] = goodfraction_systematic(n,q1,q2)
        if dic[n][0] == 1 and (n == 1 or dic[n-1][0] == 1):
            break
    for n in itertools.count(7):
        dic[n] = goodfraction_sample(n,q1,q2, m)
        if dic[n][0] == 1 and dic[n-1][0] == 1:
            break
    return dic

def is_good_graph(g,q1,q2):
    '''returns boolean True if g has a Kq1 or an Iq2'''
    #  find max clique # mc
    mc = g.omega()
    if mc >= q1: # then good graph
        #logger.debug('n={}, {}, omega={}, good'.format(n,g,mc))
        return True
    else:
        #  find max independent set # mi
        mi = g.alpha()
        if mi >= q2: # then good graph
            #logger.debug('n={}, {}, omega={}, alpha={} good'.format(n,g,mc,mi))
            return True
        else: #  bad graph!
            #logger.debug('n={}, {}, omega={}, alpha={} BAD'.format(n,g,mc,mi))
            return False

def goodfraction_systematic(n, q1, q2):
    '''Given n (<10), go through every graph of size n and count how many are good and return the fraction.
    Optimization: if q1==q2: only go through half the graphs, since the complement is the same'''
    if n < q1 and n < q2:
        return 0.0, 2 ** (n*(n-1)/2), 0.0, 0.0
    if n >= 8:
        print 'warning - this may take a while for n = ', n
    good, bad = 0, 0
    for g in generate_graphs(n, q1==q2):
        if is_good_graph(g,q1,q2):
            good += 2 if q1==q2 else 1
        else:
            bad += 2 if q1==q2 else 1
    assert good + bad == 2 ** (n*(n-1)/2)
    p = good / (good + bad)
    return p, good+bad, p, p

def generate_graphs(n, do_half):
    '''Generates all graphs of size n.
    if do_half is True, only run through the first half of the graphs.
    The second half are all complements of the first.'''
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
def goodfraction_sample(n,q1,q2, m):
    '''Sample m graphs of size n and return p, m, padj-w, padj+w
    where p is the proportion of good graphs,
    padj and w are from the confidence interval using the Wilson score method'''
##    confidence = 0.99,interval_width = 0.01
##    m = choose_sample_size(confidence,interval_width)
    good, bad = 0, 0
    for num in itertools.count(1):
        # generate random graph size n
        g = igraph.Graph.Erdos_Renyi(n=n,p=0.5)
        if is_good_graph(g,q1,q2):
            good += 1
        else:
            bad += 1
##        if num % 5000 == 0: #recalc
##            m = choose_sample_size(confidence=confidence, interval_width=interval_width,
##                                   p=good/(good+bad), df=good+bad)
        if num >= m:
            break
        if num % 500 == 0:
            pass
    # calc 99% confidence interval for proportion p
    # Use Wilson score interval: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
    # (the textbook normal approx interval fails when p close to 1)
    assert good + bad == m
    confidence = 0.99               # (parameter)
    p = good / (good+bad)
    tstar = scipy.stats.t.ppf(1- (1-confidence)/2, m)
    tstar2 = tstar**2
    padj = (p + (tstar2)/(2*m)) / (1 + (tstar2)/m)
    w = tstar * math.sqrt(p*(1-p)/m + (tstar2)/(4*(m**2))) / (1 + (tstar2)/m)
    if padj-w > p or padj+w < p:
        logger.warn('{}: p={}, padj={}, m={}, padj-w={}, padj+w={}'
                    .format(n,p,padj,m,padj-w,padj+w))
    return p, m, padj-w, padj+w

__main__()
