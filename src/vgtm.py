#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from _operator import itemgetter
import itertools
from networkx.algorithms.bipartite.centrality import degree_centrality
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.centrality import closeness_centrality
from networkx.algorithms.centrality.eigenvector import eigenvector_centrality
from networkx.algorithms.centrality.load import edge_load_centrality
from networkx.algorithms.centrality.percolation import percolation_centrality
from networkx.algorithms.centrality.second_order import second_order_centrality
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.link_analysis.hits_alg import hits
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.classes.function import degree
import pickle
import sys

import networkx as nx
import pandas as pd
from networkx.algorithms.cluster import clustering
from sklearn.decomposition import NMF
import numpy as np
from networkx.convert_matrix import to_numpy_matrix


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


def most_central_edge(G):
#    centrality = edge_load_centrality(G)
    centrality = edge_betweenness_centrality(G=G, weight='weight')
    return max(centrality, key=centrality.get)
    
    
def find_clusters(word_graph, k):
#    return asyn_fluidc(G=word_graph, k=k)
     comp = girvan_newman(word_graph, most_valuable_edge=most_central_edge)
     limited = itertools.takewhile(lambda c: len(c) <= k, comp)
     clusters = None
     printProgressBar(0, k, prefix = 'Finding clusters', suffix = 'Complete')
     for communities in limited:
         clusters = [c for c in communities]
         printProgressBar(len(clusters), k, prefix = 'Finding clusters', suffix = 'Complete')
     return clusters
    

def main(argv = None):
    
    # Command line arguments
    if argv is None:
        argv = sys.argv
    
    # Input    
    wg_fn = argv[1] # gexf or pickle file
    ds_fn = argv[2] # dataset file
    k = int(argv[3]) # number of topics
    twords = int(argv[4])
#    centrality_measure = argv[5]    # betweenness, closeness, hits, pagerank, ...
    
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
#    print("Extracting vocabulary...", end=" ", flush=True)
#    corpus_vocab = set()
#    for doc in corpus:
#        for word in doc.split():
#            corpus_vocab.add(word)
#    corpus_vocab = list(corpus_vocab)
#    corpus_vocab.sort()
#    vocab_len = len(corpus_vocab)
#    print("OK!")

    # Adjacency matrix
    wg_matrix = to_numpy_matrix(G=word_graph, nonedge=0.0001)

    # Non-negative matrix factorization
    print("NMF of adjacency matrix...", end=" ", flush=True)
    nmf_model = NMF(n_components=k, init='nndsvdar')
    nmf_model.fit_transform(wg_matrix)
    topic_word_affinity = nmf_model.components_ # matrix k x |V|
    print("OK!")

    # Calculating pwz
    pwz = []
    keys = list(word_graph.nodes())
    vocab_len = len(keys)
    printProgressBar(0, k, prefix = 'Calculating P(w|z)', suffix = 'Complete')
    for i in range(k):
        vals = list(topic_word_affinity[i])
        sum_vals = sum(vals)
        for j in range(vocab_len):
            vals[j] /= sum_vals
        pwz.append( sorted( dict(zip(keys, vals)).items(), key=lambda kv: kv[1], reverse=True ) )
        printProgressBar(i+1, k, prefix = 'Calculating P(w|z)', suffix = 'Complete')
    
    # Find clusters
#    clusters = find_clusters(word_graph, k)
    
    # Calculating pwz
#    pwz = []
#    cs = []
#    num_clusters_proc = 0
#    printProgressBar(num_clusters_proc, k, prefix = 'Calculating P(w|z)',
#                     suffix = 'Complete')
#    for c in clusters:
#        cs.append(list(c).copy())
#        # Initialization
#        pwc = dict(zip(corpus_vocab, [0.0]*vocab_len))
#        # Word importance (betweenness)
#        c_graph = word_graph.subgraph(c)
#        
#        if centrality_measure == 'betweenness':
#            nodes = betweenness_centrality(G=c_graph, normalized=False,
#                                           weight='weight')
#        elif centrality_measure == 'closeness':
#            nodes = closeness_centrality(G=c_graph, distance='distance')
#        elif centrality_measure == 'eigen':
#            nodes = eigenvector_centrality(G=c_graph, max_iter=10000, weight='weight')
#        elif centrality_measure == 'pagerank':
#            nodes = pagerank(G=c_graph)
#        elif centrality_measure == 'hits':
#            nodes = hits(G=c_graph, max_iter=10000)[1]
#        elif centrality_measure == 'secondorder':
#            nodes = second_order_centrality(G=c_graph)
#        elif centrality_measure == 'degree':
#            nodes = degree(G=c_graph, weight='weight')
#            nodes = dict(nodes)
#        elif centrality_measure == 'inverse_cc':    # Inverse of clustering coefficient
#            nodes = clustering(G=c_graph, weight='weight')   # cc
#            nodes = {k: (v+0.0001) for k, v in nodes.items()} # inverse of cc
#        
#        keys = list(nodes.keys())
#        vals = list(nodes.values())
#        sum_vals = sum(vals)
#        for i in range(len(vals)):
#            vals[i] /= sum_vals
#            if centrality_measure == 'secondorder':
#                vals[i]  = 1.0/vals[i]
#        if centrality_measure == 'secondorder':
#            sum_vals = sum(vals)
#            for i in range(len(vals)):
#                vals[i] /= sum_vals
#        nodes = dict(zip(keys, vals))
#        for n, v in nodes.items():
#            pwc[n] = v
#        pwz.append(sorted(pwc.items(), key=lambda kv: kv[1], reverse=True))
#        num_clusters_proc += 1
#        printProgressBar(num_clusters_proc, k, prefix = 'Calculating P(w|z)',
#                     suffix = 'Complete')
    
    # Salvar pwz
    pwz_file = open(pwz_fn, 'w')
    for i in range(k):
        pwz_file.write("Topic " + str(i) + "th:\n")
#         words = list(pwz[i].keys())
#         probs = list(pwz[i].values())
        for j in range(twords):
            pwz_file.write("\t" + pwz[i][j][0] + "   " + '{:.6f}'.format(pwz[i][j][1]) + "\n")
    pwz_file.close()
    
    # Salvar pz
    pz_file = open(pz_fn, 'w')
    topics_probs = topic_word_affinity.sum(axis=1)
    topics_probs /= topics_probs.sum()
#    topics_probs = []
#    topics_probs.append(len(cs[0]) / vocab_len)
    pz_file.write('{:.6f}'.format(topics_probs[0]))
    for i in range(1, k):
        pz_file.write(", " + '{:.6f}'.format(topics_probs[i]))
    pz_file.close()
    
    # Salvar pzd
    pzd_file = open(pzd_fn, 'w')
    for d in corpus:
        pzd = np.array([0.0] * k)
        for w in d.split():
            pzd += topic_word_affinity[:, keys.index(w)]
        sum_vals = sum(pzd)
        pzd[0] /= sum_vals
        pzd_file.write('{:.6f}'.format(pzd[0]))
        for i in range(1,k):
            pzd[i] /= sum_vals
            pzd_file.write(' ' + '{:.6f}'.format(pzd[i]))
        pzd_file.write("\n")
    pzd_file.close()

#    word_topics = dict()
#    for i in range(k):
#        for w in cs[i]:
#            word_topics[w] = i
#    pzd_file = open(pzd_fn, 'w')
#    for d in corpus:
#        pzd = [0.001] * k
#        for w in d.split():
#            try:
#                wtopic = word_topics[w]
#                pzd[wtopic] += 1
#            except:
#                pass
#        sum_vals = sum(pzd)
#        pzd[0] /= sum_vals
#        pzd_file.write('{:.6f}'.format(pzd[0]))
#        for i in range(1,k):
#            pzd[i] /= sum_vals
#            pzd_file.write(' ' + '{:.6f}'.format(pzd[i]))
#        pzd_file.write("\n")
#    pzd_file.close()

if __name__ == '__main__':
    main()
