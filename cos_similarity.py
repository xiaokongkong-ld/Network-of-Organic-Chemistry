import os
import random

import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

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
from tqdm import tqdm
from tensorflow import keras
from sklearn.linear_model import Ridge
from sklearn.svm import SVC


DATA_PATH = "D:\\fingerPrintIndexedLinks.txt"
FINGERPRINT = "D:\\compoundsWithFingerprintMatrix.txt"

def load_data(data_path,fingerprint_path, num_rows):

    print("data loading...")
    dtype = {"reactantID": np.int32, "productID": np.int32}

    reactions = pd.read_table(data_path, sep=' ', dtype=dtype, usecols=range(2), nrows=num_rows)
    fingerprint = pd.read_table(fingerprint_path, sep=' ', dtype=dtype, usecols=range(168), nrows=num_rows * 2)
    compound_list = list(fingerprint["compoundID"])
    fingerprint.set_axis(compound_list, inplace=True)


    reactions["reactions"] = 1
    print("data loaded")
    print("building pivot table")
    reaction_matrix = reactions.pivot_table(index=["reactantID"], columns=["productID"], values="reactions")

    reaction_matrix.fillna(0, inplace=True)

    reactant_list = list(reaction_matrix.index)
    product_list = list(reaction_matrix.columns)

    fp = fingerprint.loc[reactant_list]
    fp.drop(['compoundID'], axis=1, inplace=True)
    reaction_withfingerprint_matrix = pd.concat([reaction_matrix, fp], axis=1)

    return reactions, reaction_matrix, reactant_list, product_list, reaction_withfingerprint_matrix

reactions_all, reaction_matrix, reactant_list, product_list, rea_wit_fin_mat = load_data(DATA_PATH, FINGERPRINT,5000)
print(".............reaction with fingerprint matrix...............")
print(rea_wit_fin_mat)
print("...........calculating similarity with fingerprint..........")

print("calculating Pearson correlation coefficient...")
#similarity = reaction_matrix.T.corr()
cos_similarity = pd.DataFrame(cosine_similarity(rea_wit_fin_mat))

cos_similarity.set_axis((reactant_list), inplace=True)
cos_ine_similarity = cos_similarity.T
cos_ine_similarity.set_axis((reactant_list), inplace=True)

print(reaction_matrix)
print(cos_ine_similarity)

print(reactions_all)

def predict(reactantID, productID, reaction_matrix, similarity, K):

    similar_reactants = similarity[reactantID].drop([reactantID]).dropna()
    similar_reactants = similar_reactants.where(similar_reactants>0).dropna()
    if similar_reactants.empty is True:
        raise Exception("reactant <%d> has no similar reactant" %reactantID)

    similar_reactants = similar_reactants.sort_values(ascending=False)
    top_K_similar_reactants = similar_reactants[:K]

    print("Top %d" %K + " similar reactants of reactant %d:" %reactantID)
    print(top_K_similar_reactants)

    sum_up = 0
    sum_down = 0
    print("top %d similar reactant" % K + " to product %d" % productID)
    for sim_reID, similaritys in top_K_similar_reactants.iteritems():
        sim_reactant_reaction = reaction_matrix.loc[sim_reID].dropna()
        connection_reactant_product = sim_reactant_reaction.loc[productID]

        print(connection_reactant_product)

        sum_up += similaritys*connection_reactant_product
        sum_down += similaritys

    predict_connection = sum_up/sum_down
    #if predict_connection<=0.5:
     #   predict_connection = 0.0
  #  else:
     #   predict_connection = 1.0

    print("Probability of reactant %d to product %d is %f" %(reactantID, productID, predict_connection))
    print(type(predict_connection))
    return predict_connection

def get_connected_samples(re_matrx, reaction_links, n_pos_samples):
    reaction_links = pd.DataFrame(reaction_links)
    sample_pos = pd.DataFrame.sample(reaction_links, 1)
    i = 0
    while i<n_pos_samples:
        pos_sample = pd.DataFrame.sample(reaction_links, 1)
        temp = pos_sample.reset_index()
        id = temp.at[0,"reactantID"]
        id_row_at_matrix = pd.DataFrame(re_matrx.loc[id])
        row_sum = id_row_at_matrix.sum(axis = 0)
        if int(row_sum)>1:
            sample_pos = sample_pos.append(pos_sample)
            i = i+1

    sample_pos = sample_pos.reset_index()
    sample_pos = sample_pos.drop([0])

    return sample_pos

def random_sample_predict(reaction_list, reaction_matrix, reactant_list, product_list, similar, N_nega_samples, top_K_simlar_reactants):

    y = []
    y_ = []


    #positive samples

    N_pos_samples = N_nega_samples*N_nega_samples

    connected_pairs = get_connected_samples(reaction_matrix, reaction_list, N_pos_samples)
    print(connected_pairs)
    for i in range(N_pos_samples):
        reactant_id = connected_pairs.at[i + 1,"reactantID"]
        product_id = connected_pairs.at[i + 1, "productID"]
        y.append(predict(reactant_id, product_id, reaction_matrix, similar, top_K_simlar_reactants))
        y_.append(1.0)

    # negative samples

    sample_row = random.sample(reactant_list, N_nega_samples)
    sample_col = random.sample(product_list, N_nega_samples)

    sample_matrix = reaction_matrix.loc[sample_row]
    sample_matrix = sample_matrix[sample_col]
    for i in sample_row:
        for j in sample_col:
            y.append(predict(i, j, reaction_matrix, similar, top_K_simlar_reactants))
            y_.append(sample_matrix.loc[i][j])

    return y, y_

def accuracy_score(predict, actual):
    lens = len(predict)
    up = 0
    down = 0
    for i in range(lens):
        if actual[i]==1:
            up = up + predict[i]
            down = down + actual[i]
        else:
            if predict[i] == 0:
                up = up + 1
                down = down + 1
            else:
                up = up - predict[i] + 1
                down = down + 1

    score = up / down
    return score


print("..............random sample score....................")

predict_connection_prob, actual = random_sample_predict(reactions_all, reaction_matrix, reactant_list, product_list, cos_ine_similarity, 2, 2)

#pre = np.trunc(predict_connection_prob).astype(float).tolist()
#act = np.trunc(actual).astype(float).tolist()
pre = predict_connection_prob
act = actual

print("predict probability:")
print(pre)
print("actual connection:")

print(act)
print("accuracy score:")
#ac_score = sklearn.metrics.accuracy_score(act, pre, normalize=True, sample_weight=None)
ac_score = accuracy_score(pre,act)
print(ac_score)

#false_positive_rate, true_positive_rate, thresholds = roc_curve(act, pre, pos_label=None, sample_weight=None, drop_intermediate=True)
#auc_score=auc(false_positive_rate, true_positive_rate)

#print("AUC score:")
#print(auc_score)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__

accuracy = []
MAX_ITER = 20
print("\n***************************** Compute Accuracy  ****************************")
for i in (range(MAX_ITER)):
    print("\nITERATION ", i + 1, "/", MAX_ITER)
    blockPrint()
    predict_connection_prob, actual = random_sample_predict(reactions_all, reaction_matrix, reactant_list, product_list,
                                                            cos_ine_similarity, 2, 5)
    pre = predict_connection_prob
    act = actual
    score = accuracy_score(pre,act)
    accuracy.append(score)
    enablePrint()
    print("number ", i + 1, "accuracy score is", score)

accuracy = np.array(accuracy)
avg_acc = np.mean(accuracy)

enablePrint()
print("\n***************************** Compute Accuracy DONE *****************************")
print("\nOn average, the accuracy is: ", avg_acc)