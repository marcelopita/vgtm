#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import networkx as nx
import pickle
import pandas as pd


def printProgressBar (iteration, total, prefix = '', suffix = '',
                      decimals = 1, length = 50, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '░' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
        
        
def load_dataset(dataset_filename):
    return pd.read_csv(dataset_filename, sep=";",
                       header=None,
                       names=["id", "class", "text"],
                       index_col=0)


def main(argv = None):
    
    # Command line arguments
    if argv is None:
        argv = sys.argv
    
    # Input    
    wg_fn = argv[1] # gexf or pickle file
    ds_fn = argv[2] # dataset file
    centrality_measure = argv[3]    # betweenness, closeness, hits, pagerank, ...
    
    # Output
    pwz_fn = argv[4]    # word-topic probabilities file
    pz_fn = argv[5]     # topic probabilities file
    pzd_fn = argv[6]    # topic-document probabilities file
    
    # Load word graph
    print("Loading word graph...", end = " ", flush = True)
    word_graph = None
    if wg_fn.endswith(".gexf"):
        word_graph = nx.read_gexf(wg_fn)
    elif wg_fn.endswith(".pkl"):
        with open(wg_fn, "r") as wg_file:
            word_graph = pickle.load(wg_file)
    print("OK!")
    
    # Load dataset
    print("Loading corpus...", end=" ", flush=True)
    corpus = None
    if ds_fn.endswith(".txt"):
        with open(ds_fn, "r") as ds_file:
            for line in ds_file:
                if corpus is None:
                    corpus = []
                corpus.append(line.split())
    elif ds_fn.endswith(".csv"):
        ds = load_dataset(ds_fn)
        corpus = ds['text'].tolist()
    print("OK!")


if __name__ == '__main__':
    main()