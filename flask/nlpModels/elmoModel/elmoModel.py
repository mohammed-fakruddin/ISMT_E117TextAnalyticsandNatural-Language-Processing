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


def get_elmo_scores(search_query, elmoVectorsForFundObjectives):
    #pickle_in = open("elmo_vectors_03122019.pickle", "rb")
    #elmoVectorsForFundObjectives = pickle.load(pickle_in)
    url = "https://tfhub.dev/google/elmo/2"

    g = tf.Graph()
    with g.as_default():
        #text_input = tf.placeholder(dtype=tf.string, shape=[None])
        #embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        #embed = hub.Module(url)
        #embed = elmoModelPreTrained
        #my_result = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()


    elmoModelPreTrained = hub.Module(url)
    search_string_pre_process = preProcess.pre_process_sent(search_query)[0]
    search_string_pre_process = " ".join(search_string_pre_process)
    embeddings2 = elmoModelPreTrained([search_string_pre_process],signature="default",as_dict=True)["default"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        search_vect = sess.run(embeddings2)

    cosine_similarities = pd.Series(cosine_similarity(search_vect, elmoVectorsForFundObjectives).flatten())
    scores=[]
    for idx,score in cosine_similarities.nlargest(int(5)).iteritems():
        scores.append([idx,score])
    return scores
