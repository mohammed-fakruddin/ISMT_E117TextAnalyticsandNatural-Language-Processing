from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import string
import operator

import pickle

from sklearn.metrics.pairwise import cosine_similarity

from pythonModules import preProcess

def embed_texts(texts):

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(texts))

    return np.array(embeddings).tolist()

def get_use_scores(search_query, useVectorsForFundObjectives):
    #pickle_in = open("elmo_vectors_03122019.pickle", "rb")
    #elmoVectorsForFundObjectives = pickle.load(pickle_in)
    url = "https://tfhub.dev/google/universal-sentence-encoder/2"

    g = tf.Graph()
    with g.as_default():
        #text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module(url)
        #embed = hub.Module(url)
        #embed = elmoModelPreTrained
        #my_result = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()


    useModelPreTrained = hub.Module(url)
    search_string_pre_process = preProcess.pre_process_sent(search_query)[0]
    search_string_pre_process = " ".join(search_string_pre_process)

    embeddings2 = useModelPreTrained([search_string_pre_process])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        search_vect = sess.run(embeddings2)



    score  = np.inner(search_vect, useVectorsForFundObjectives)
    #sort the records in descending order of scores and pick the top 5
    topk_idx = np.argsort(-score)[::-1][0][:5]
    #print(topk_idx)
    scores=[]
    for idx in topk_idx:
        #print(idx)
        #print('> %s\t%s' % (score[0][idx], fund_objectives[idx]))
        print('USE Model scores', score[0][idx])
        scores.append([idx,score[0][idx]])

    return scores
