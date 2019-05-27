#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import networkx as nx
import pickle
import pandas as pd
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.centrality import betweenness_centrality
import itertools


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
    
    
def find_clusters(word_graph, k):
    comp = girvan_newman(word_graph)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    clusters = None
    printProgressBar(0, k, prefix = 'Finding clusters',
                     suffix = 'Complete')
    for communities in limited:
        clusters = [c for c in communities]
        printProgressBar(len(clusters), k, prefix = 'Finding clusters',
                         suffix = 'Complete')
    return clusters
    


def main(argv = None):
    
    # Command line arguments
    if argv is None:
        argv = sys.argv
    
    # Input    
    wg_fn = argv[1] # gexf or pickle file
    ds_fn = argv[2] # dataset file
    k = int(argv[3]) # number of topics
    centrality_measure = argv[4]    # betweenness, closeness, hits, pagerank, ...
    
    # Output
    pwz_fn = argv[5]    # word-topic probabilities file
    pz_fn = argv[6]     # topic probabilities file
    pzd_fn = argv[7]    # topic-document probabilities file
    
    # Load word graph
    print("Loading word graph...", end = " ", flush = True)
    word_graph = None
    if wg_fn.endswith(".gexf"):
        word_graph = nx.read_gexf(wg_fn)
    elif wg_fn.endswith(".pkl"):
        with open(wg_fn, "rb") as wg_file:
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
                corpus.append(line.strip())
    elif ds_fn.endswith(".csv"):
        ds = load_dataset(ds_fn)
        corpus = ds['text'].tolist()
    print("OK!")
    
    # Sorted vocabulary
    print("Extracting vocabulary...", end=" ", flush=True)
    corpus_vocab = set()
    for doc in corpus:
        for word in doc.split():
            corpus_vocab.add(word)
    corpus_vocab = list(corpus_vocab)
    corpus_vocab.sort()
    vocab_len = len(corpus_vocab)
    print("OK!")
    
    # Find clusters
    clusters = find_clusters(word_graph, k)
    
    # Calculating pwz
    pwz = []
    num_clusters_proc = 0
    printProgressBar(num_clusters_proc, k, prefix = 'Calculating P(w|z)',
                     suffix = 'Complete')
    for c in clusters:
        # Initialization
        pwc = dict(zip(corpus_vocab, [0.0]*vocab_len))
        # Word importance (betweenness)
        c_graph = word_graph.subgraph(c)
        nodes = betweenness_centrality(G=c_graph, normalized=False,
                                       weight='weight')
        keys = list(nodes.keys())
        vals = list(nodes.values())
        sum_vals = sum(vals)
        for i in range(len(vals)):
            vals[i] /= sum_vals
        nodes = dict(zip(keys, vals))
        for n, v in nodes.items():
            pwc[n] = v
        pwz.append(sorted(pwc.items(), key=lambda kv: kv[1], reverse=True))
        num_clusters_proc += 1
        printProgressBar(num_clusters_proc, k, prefix = 'Calculating P(w|z)',
                     suffix = 'Complete')
    
    # Salvar pwz


if __name__ == '__main__':
    main()