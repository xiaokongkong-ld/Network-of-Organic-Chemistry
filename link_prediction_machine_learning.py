import random
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import sklearn
import networkx as nx
from sklearn.linear_model import LinearRegression
from node2vec import Node2Vec
import sys, os
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm
from tensorflow import keras
from sklearn.linear_model import Ridge
from sklearn.svm import SVC


class LinkPred():
    def get_unconnected_nodes(G, source, target):

        # combine all entities in a list
        node_list = source + target

        # remove dups from list
        node_list = list(dict.fromkeys(node_list))

        # build adj matrix
        adj_G = nx.to_numpy_matrix(G, nodelist=node_list)

        unconnected_pairs = []
        offset = 0

        for i in (range(adj_G.shape[0])):
            for j in range(offset, adj_G.shape[1]):
                if i != j:
                    if adj_G[i, j] == 0:
                        unconnected_pairs.append([node_list[i], node_list[j]])

            offset = offset + 1

        # negative samples during training of the model
        node1_unlinked = [i[0] for i in unconnected_pairs]
        node2_unlinked = [i[1] for i in unconnected_pairs]

        return unconnected_pairs, node1_unlinked, node2_unlinked

    def get_removable_links(G, kg_df):

        node_nb = len(G.nodes)
        kg_df_temp = kg_df.copy()

        removable_links = []

        for i in (kg_df.index.values):
            # remove pair and build a new graph
            G_temp = nx.from_pandas_edgelist(kg_df_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph)

            # check nb of connect comp >1 && node_nb is the same
            if (nx.number_connected_components(G_temp) > 1 and len(G_temp.nodes) == node_nb):
                removable_links.append(i)
                kg_df_temp = kg_df_temp.drop(index=i)

        return removable_links

    def predict(nb_dropped_links):

        pairs = []

        x, y = np.genfromtxt('D:\\fingerPrintIndexedLinks.txt', dtype=int, unpack=True)
        for i in range(100):
            pairs.append([x[i], y[i]])
        pairs = np.array(pairs)

        nodes1 = [i[0] for i in pairs]
        nodes2 = [i[1] for i in pairs]

        kg_df = pd.DataFrame({'node_1': nodes1, 'node_2': nodes2})

        print("\ncreate data frame of all links\n")
        print("\ndata frame of all linked node pairs: \n", kg_df.head())
        print("\nshape of linked node pairs: ", kg_df.shape)

        G = nx.from_pandas_edgelist(kg_df, "node_1", "node_2", create_using=nx.Graph())
        print("\nLink nodes graph info:\n ", nx.info(G))


        # Get unconnected pair samples
        unconnected_pairs, node1_unlinked, node2_unlinked = LinkPred.get_unconnected_nodes(G, nodes1, nodes2)


        data = pd.DataFrame({'node_1': node1_unlinked, 'node_2': node2_unlinked})
        data['link'] = 0
        data_temp=data.copy()

        nega=pd.DataFrame.sample(data_temp, n=nb_dropped_links, frac=None, replace=False, weights=None, random_state=None, axis=0)


        print("\ndata frame of unlinked node pairs: \n", data.head())
        print("\ndata frame shape of unlinked node pairs: ", data.shape)

        removable_links = LinkPred.get_removable_links(G, kg_df)
        print("\nremovable links: ", removable_links)
        # Append removable edges to dataframe of unconnected node pairs

        # Remove 3 edges
        links = random.sample(removable_links, nb_dropped_links)

        kg_df_ghost = kg_df.loc[links]
        kg_df_ghost['link'] = 1

        print("\nremoved link samples: \n", kg_df_ghost.head())
        print("\ndata frame of removed links: ", kg_df_ghost.shape)

        # Drop all removable links
        kg_df_partial = kg_df.drop(index=kg_df_ghost.index.values)
        kg_df_partial['link'] = 1

        data = data.append(kg_df_partial[['node_1', 'node_2', 'link']], ignore_index=True)

        kg_df_ghost = kg_df_ghost.append(nega[['node_1', 'node_2', 'link']], ignore_index=True)

        print("\nfull model data frame\n", data.head())
        print("\ndata shape of the model: ", data.shape)

        # Make a new graph missing all the removable links
        G_data = nx.from_pandas_edgelist(kg_df_partial, "node_1", "node_2", create_using=nx.Graph)
        print("\ninformation of graph after drop linked pairs:\n ", nx.info(G_data))
        print('\n')


        node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50,p=4,q=0.1)
        n2w_model = node2vec.fit(window=7, min_count=1)


        x = [(n2w_model.wv.__getitem__(str(i)) + n2w_model.wv.__getitem__(str(j))) for i, j in
             zip(data['node_1'], data['node_2'])]
        x_ = [(n2w_model.wv.__getitem__(str(i)) + n2w_model.wv.__getitem__(str(j))) for i, j in
              zip(kg_df_ghost['node_1'], kg_df_ghost['node_2'])]

        x_train_3 = np.array(x)
        y_train_3 = data['link']

        x_test_3 = np.array(x_)
        y_test_3 = kg_df_ghost['link']

        #change machine learning model: logistic regression and SVC

        #lr = sklearn.linear_model.LogisticRegression(class_weight="balanced", max_iter=600)
        #lr.fit(x_train_3, y_train_3)
        #predictions = lr.predict(x_test_3)

        model = SVC(kernel='rbf')
        model.fit(x_train_3, y_train_3)
        predictions = model.predict(x_test_3)


        print("\n links   : \n", kg_df_ghost)
        print("\nexpected : \n", np.array(y_test_3[:]))
        print("\nresult   : \n", predictions[:])
        score_acc = sklearn.metrics.accuracy_score(y_test_3, predictions, normalize=True, sample_weight=None)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_3, predictions, pos_label=None, sample_weight=None, drop_intermediate=True)
        score_auc=auc(false_positive_rate, true_positive_rate)
        score_f1 = sklearn.metrics.f1_score(y_test_3, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        score_rec = sklearn.metrics.recall_score(y_test_3, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        score_pre = sklearn.metrics.precision_score(y_test_3, predictions, labels=None, pos_label=1, average='binary')


        return score_acc, score_auc, score_f1, score_rec, score_pre

    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    def enablePrint(self):
        sys.stdout = sys.__stdout__


pass

if __name__ == '__main__':
    test = LinkPred()

    score = LinkPred.predict(9)

    print("\n***************************** Compute Accuracy  ****************************")

    accuracy = []
    sc_auc = []
    sc_f1 = []
    sc_rec = []
    sc_pre = []
    MAX_ITER = 10

    for i in (range(MAX_ITER)):
        print("\niteration ", i + 1, "/", MAX_ITER)

        test.blockPrint()
        score_acc, score_auc, score_f1, score_rec, score_pre = LinkPred.predict(3)
        accuracy.append(score_acc)
        sc_auc.append(score_auc)
        sc_f1.append(score_f1)
        sc_rec.append(score_rec)
        sc_pre.append(score_pre)
        test.enablePrint()
        print("number ", i + 1, "accuracy score is", score_auc)
        print("number ", i + 1, "AUC score is", score_auc)
        print("number ", i + 1, "F1 score is", score_f1)
        print("number ", i + 1, "Recall score is", score_rec)
        print("number ", i + 1, "Precision score is", score_pre)


    accuracy = np.array(accuracy)
    avg_acc = np.mean(accuracy)

    sc_auc = np.array(sc_auc)
    avg_auc = np.mean(sc_auc)

    sc_f1 = np.array(sc_f1)
    avg_f1 = np.mean(sc_f1)

    sc_rec = np.array(sc_rec)
    avg_rec = np.mean(sc_rec)

    sc_pre = np.array(sc_pre)
    avg_pre = np.mean(sc_pre)

    test.enablePrint()

    print("\nOn average, the accuracy score is: ", avg_acc)
    print("\nOn average, the AUC score is: ", avg_auc)
    print("\nOn average, the F1-score is: ", avg_f1)
    print("\nOn average, the Recall score is: ", avg_rec)
    print("\nOn average, the Precision is: ", avg_pre)
