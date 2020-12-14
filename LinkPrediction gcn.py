# coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from tf_geometric.utils.graph_utils import edge_train_test_split, negative_sampling

DATA_PATH = "D:\\DATA/fingerPrintIndexedLinks.txt"
FINGERPRINT = "D:\\DATA/compoundsWithFingerprintMatrix.txt"

def load_data(data_path,fingerprint_path, num_rows):

    print("data loading...")
    dtype = {"reactantID": np.int64, "productID": np.int64}

    reactions = pd.read_table(data_path, sep=' ', dtype=dtype, usecols=range(2), nrows=num_rows)
    max_row = reactions.max(axis=1)
    max = max_row.max(axis=0)

    fingerprint = pd.read_table(fingerprint_path, sep=' ', dtype=dtype, usecols=range(168), nrows=max)
    fingerprint = fingerprint.drop(columns = ['compoundID'])

    reactions = np.array(reactions.T)
    a = np.ones(shape=reactions.shape)
    reactions = reactions - a
    reactions = pd.DataFrame(reactions, dtype=np.int64)
    fingerprint = np.array(fingerprint)
    reactions = np.array(reactions)


    return reactions, fingerprint

reactions_all, compounds_fp = load_data(DATA_PATH, FINGERPRINT,1000)

graph = tfg.Graph(
    x=compounds_fp,
    edge_index=reactions_all

)

# undirected edges can be used for evaluation
undirected_train_edge_index, undirected_test_edge_index, _, _ = edge_train_test_split(
    edge_index=graph.edge_index,
    test_size=0.15
)

# use negative_sampling with replace=False to create negative edges for test
undirected_test_neg_edge_index = negative_sampling(
    num_samples=undirected_test_edge_index.shape[1],
    num_nodes=graph.num_nodes,
    edge_index=graph.edge_index,
    replace=False
)

# convert undirected edges to directed edges for correct GCN propagation
train_graph = tfg.Graph(x=graph.x, edge_index=undirected_train_edge_index).convert_edge_to_directed()


embedding_size = 16
drop_rate = 0.3

gcn0 = tfg.layers.GCN(32, activation=tf.nn.relu)
gcn1 = tfg.layers.GCN(embedding_size)
dropout = keras.layers.Dropout(drop_rate)


def encode(graph, training=False):
    h = gcn0([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
    h = dropout(h, training=training)
    h = gcn1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
    return h


def predict_edge(embedded, edge_index):
    row, col = edge_index
    embedded_row = tf.gather(embedded, row)
    embedded_col = tf.gather(embedded, col)

    # dot product
    logits = tf.reduce_sum(embedded_row * embedded_col, axis=-1)
    return logits


def compute_loss(pos_edge_logits, neg_edge_logits):
    pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=pos_edge_logits,
        labels=tf.ones_like(pos_edge_logits)
    )

    neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=neg_edge_logits,
        labels=tf.zeros_like(neg_edge_logits)
    )

    return tf.reduce_mean(pos_losses) + tf.reduce_mean(neg_losses)


def evaluate():
    embedded = encode(train_graph)

    pos_edge_logits = predict_edge(embedded, undirected_test_edge_index)
    neg_edge_logits = predict_edge(embedded, undirected_test_neg_edge_index)

    pos_edge_scores = tf.nn.sigmoid(pos_edge_logits)
    neg_edge_scores = tf.nn.sigmoid(neg_edge_logits)

    y_true = tf.concat([tf.ones_like(pos_edge_scores), tf.zeros_like(neg_edge_scores)], axis=0)
    y_pred = tf.concat([pos_edge_scores, neg_edge_scores], axis=0)

    #auc_m = keras.metrics.AUC()
    auc_m = keras.metrics.Precision()
    auc_m = keras.metrics.Recall()
    auc_m.update_state(y_true, y_pred)

    return auc_m.result().numpy()


optimizer = tf.optimizers.Adam(learning_rate=1e-2)
for step in range(1000):
    with tf.GradientTape() as tape:
        embedded = encode(train_graph, training=True)

        # negative sampling for training
        train_neg_edge_index = negative_sampling(
            train_graph.num_edges,
            graph.num_nodes,
            edge_index=train_graph.edge_index
        )

        pos_edge_logits = predict_edge(embedded, train_graph.edge_index)
        neg_edge_logits = predict_edge(embedded, train_neg_edge_index)

        loss = compute_loss(pos_edge_logits, neg_edge_logits)

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        auc_score = evaluate()
        print("step = {}\tloss = {}\tauc_score = {}".format(step, loss, auc_score))

