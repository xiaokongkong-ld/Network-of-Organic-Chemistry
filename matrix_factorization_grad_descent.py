# -*- coding: utf-8 -*-
import time
import os
import random

import numpy
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
import numpy as np

DATA_PATH = "D:\\fingerPrintIndexedLinks.txt"
CHACHE_DIR = "D:\\"

def load_data(data_path):
    cache_path = os.path.join(CHACHE_DIR, "reactions_matrix.cache")
    print("data loading...")
    dtype = {"reactantID": np.int32, "productID": np.int32}

    reactions = pd.read_table(data_path, sep=' ', dtype=dtype, usecols=range(2), nrows=500)
    reactions["reactions"] = 1
    print("data loaded")
    print("building pivot table")
    reaction_matrix = reactions.pivot_table(index=["reactantID"], columns=["productID"], values="reactions")
    print("pivot table done")
    #reaction_matrix.fillna(0, inplace=True)

    reactant_list = list(reaction_matrix.index)
    product_list = list(reaction_matrix.columns)

    return reactions, reaction_matrix, reactant_list, product_list

reactions_all, reaction_matrix, reactant_list, product_list = load_data(DATA_PATH)

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
connected_pairs = get_connected_samples(reaction_matrix, reactions_all, 10)
print("........connections to be removed.........")
print(connected_pairs)

print(".........original reaction matrix.........")
print(reaction_matrix)

for i in range(len(connected_pairs)):
    reaction_matrix.at[connected_pairs.at[i+1, "reactantID"], connected_pairs.at[i+1, "productID"]] = "NaN"
print(".........reaction matrix remove some links.........")
print(reaction_matrix)


def matrix_factorization(R,P,Q,d,steps,alpha=0.05,lamda=0.002):
    Q=Q.T
    sum_st = 0
    e_old = 0
    flag = 1
    for step in range(steps):
        st = time.time()
        e_new = 0
        for u in range(len(R)):
            for i in range(len(R[u])):
                if R[u][i]>0:
                    eui=R[u][i]-np.dot(P[u,:],Q[:,i])
                    for k in range(d):
                        P[u][k] = P[u][k] + alpha*eui * Q[k][i]- lamda *P[u][k]
                        Q[k][i] = Q[k][i] + alpha*eui * P[u][k]- lamda *Q[k][i]
        cnt = 0
        for u in range(len(R)):
            for i in range(len(R[u])):
                if R[u][i]>0:
                    cnt = cnt + 1
                    e_new = e_new + pow(R[u][i]-np.dot(P[u,:],Q[:,i]),2)
        et = time.time()
        e_new = e_new / cnt

        print(step,";",e_old)

        if step == 0:
            e_old = e_new
            continue
        sum_st = sum_st + (et-st)
        if e_new<1e-3:
            flag = 2
            break
        if e_old - e_new<1e-10:
            flag = 3
            break
        else:
            e_old = e_new
    print('---------Summary----------\n',
      'Type of jump out:',flag,'\n',
      'Total steps:',step + 1,'\n',
      'Total time:',sum_st,'\n',
      'Average time:',sum_st/(step+1.0),'\n',
      "The e is:",e_new)
    return P,Q.T

re_ma = reaction_matrix.reset_index(drop = True)
re_ma = re_ma.T.reset_index(drop=True).T

R=re_ma.values

d = 8
steps = 100
N = len(R)
M = len(R[0])
P = np.random.normal(loc=0,scale=0.01,size=(N,d))
Q = np.random.normal(loc=0,scale=0.01,size=(M,d))
nP,nQ = matrix_factorization(R,P,Q,d,steps)

print('.......approximate matrix of reaction matrix.......')
predict_matrix = pd.DataFrame(np.dot(nP,nQ.T))
predict_matrix.set_axis((reactant_list), inplace=True)
predict_matrix = predict_matrix.T
predict_matrix.set_axis((product_list),inplace=True)
predict_matrix = predict_matrix.T
print(predict_matrix)

y = []
y_ = []
pre = []
act = []

for i in range(len(connected_pairs)):
    predict_prob = predict_matrix.at[connected_pairs.at[i+1, "reactantID"], connected_pairs.at[i+1, "productID"]]
    if predict_prob>=0.5:
        predict_prob = 1.0
    else:
        predict_prob = 0.0
    y.append(predict_prob)
    y_.append(1.0)



print("prediction:")
print(y)
print("actual")
print(y_)

def accuracy_score(predict, actual):
    up = 0
    down = 0
    for i in range(len(predict)):
        up = up + predict[i]
        down = down + actual[i]
    score = up/down
    return score

ac_score = accuracy_score(y, y_)
print("accuracy score:")
print(ac_score)

