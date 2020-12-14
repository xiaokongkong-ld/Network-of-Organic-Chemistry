from __future__ import print_function
import networkx as nx
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from math import log

##################################
######### READ EDGE LIST #########
##################################

print('Reading edgelist')
# Read combined edge-list
os.chdir('D:\\')
filename = 'fingerPrintIndexedLinks_.txt'
#filename = 'test_.txt'

compounList = []
reactionList = []
G = nx.DiGraph()
de = G.degree
edges_f = open(filename, 'rb')
with open(filename, encoding='UTF-8') as file:
    i = 0
    for line in file:
        if i<500:

            reactant, product = [str(x) for x in line.split(" ")]
            G.add_node(reactant, name=str(reactant))
            G.add_node(product, name=str(product))
            G.add_edge(reactant, product)
        i = i+1

name= nx.get_node_attributes(G, 'name')
edges_f = open(filename, 'rb')
# Parse edgelist into directed graph
noc_g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph())
print('Num. weakly connected components: ', nx.number_weakly_connected_components(G))
print('Saving adjacency matrix')

# Get adjacency matrix
adj = nx.adjacency_matrix(noc_g)

# Save adjacency matrix
with open('D:\\fingerPrintIndexedLinks.pkl', 'wb') as f:
    pickle.dump(adj, f)


##################################
##### VISUALIZATIONS, STATS ######
##################################

# Generate visualization
def save_visualization(g, file_name):
    plt.figure(figsize=(18, 18))
    degrees = g.degree
    name = nx.get_node_attributes(g, 'name')
    nn = [i[1] for i in degrees]
    nl = [i[0] for i in degrees]
    # Draw networkx graph -- scale node size by log(degree+1)
    nx.draw_spring(g, with_labels=True,
                   linewidths=0.5,
                   edge_color='green',
                   labels=name,
                   nodelist=nl,
                   node_size=[log(degree_val + 1) * 200 for degree_val in nn],
                   node_color=[log(degree_vall + 1) * 5000 for degree_vall in nn],
                   arrows=True)

    # Create black border around node shapes
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")

    #     plt.title(title)
    plt.savefig(file_name)
    plt.clf()


# Save network stats to .txt file
def save_network_statistics(g):
    degrees = g.degree

    nn = [i[1] for i in degrees]

    stats = {}
    stats['num_weakly_connected_components'] = nx.number_weakly_connected_components(g)
    stats['num_strongly_connected_components'] = nx.number_strongly_connected_components(g)
    stats['num_nodes'] = nx.number_of_nodes(g)
    stats['num_edges'] = nx.number_of_edges(g)
    stats['density'] = nx.density(g)
    try:
        stats['avg_clustering_coef'] = nx.average_clustering(g)
    except:
        stats['avg_clustering_coef'] = None  # not defined for directed graphs
    stats['avg_degree'] = sum(nn) / float(stats['num_nodes'])
    stats['transitivity'] = nx.transitivity(g)
    try:
        stats['diameter'] = nx.diameter(g)
    except:
        stats['diameter'] = None  # unconnected --> infinite path length between connected components

    with open('D:\\fingerprint-statistics-undirected.txt', 'w') as f:
        for stat_name, stat_value in stats.items():
            temp = str(stat_value)
            f.write(stat_name + ': ' + temp + '\n')


#print('Generating network visualization')
#save_visualization(g=G, file_name='D:\\fingerPrintIndexedLinks-visualization.pdf')

print('Calculating and saving network statistics')
print(nx.average_clustering(noc_g))
save_network_statistics(G)