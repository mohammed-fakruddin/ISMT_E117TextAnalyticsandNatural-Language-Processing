from flask import Flask, render_template, request
import requests
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import string
import numpy as np
import nltk
import spacy
import gensim
from gensim import models
from gensim import similarities
from gensim import corpora
from gensim.models.fasttext import FastText
import matplotlib.pyplot as plt
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import pickle

from bert_serving.client import BertClient




#=================
# custom python modules
#=================
from pythonModules import preProcess
from nlpModels.word2VecModel import word2VecModel
from nlpModels.elmoModel import elmoModel
from nlpModels.useModel import useModel
from nlpModels.lsiModel import lsiModel
from nlpModels.tfIdfModel import tfIdfModel
from nlpModels.bertModel import bertModel
from nlpModels.fastTextModel import fastTextModel

#### initialize the flask app.

app = Flask(__name__)

################################
# home page
################################
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    errors=[]
    results=[]
    top_scores={"LSI":[], "Word2Vec":[], "USE":[], "Glove":[], "BERT":[], "ELMO":[], "TFIDF":[], "FastText":[]}
    top_objectives={"LSI":[], "Word2Vec":[], "USE":[], "Glove":[], "BERT":[], "ELMO":[], "TFIDF":[], "FastText":[]}
    top_docIdx={"LSI":[], "Word2Vec":[], "USE":[], "Glove":[], "BERT":[], "ELMO":[], "TFIDF":[], "FastText":[]}
    message=""
    query_string=""
    topFunds=""
    if request.method == "POST":
        sims={"Cosine":[0.96,0.90,0.99]}
        query_string = request.form.get('message')
        print('search query string', query_string)
        #scores = nlp_processing(message)
        #
        #####################
        #word 2 vec model call
        print('===============Word2VecModel ==============', query_string)
        w2vScores = word2VecModel.get_word2Vec_scores(query_string, w2vModel)
        top_scores["Word2Vec"] = [round(x[1],4) for x in w2vScores]
        w2vDocIdx = [x[0] for x in w2vScores]
        top_objectives["Word2Vec"]=[fund_objectives[int(idx)] for idx in w2vDocIdx]
        top_docIdx["Word2Vec"] = [int(idx) for idx in w2vDocIdx]
        #####################
        # ELMO Model
        #elmoScores = elmoModel.get_elmo_scores(query_string, elmoVectorsForFundObjectives, elmoModelPreTrained, graph, session)
        print('===============ElmoModel ==============', query_string)
        elmoScores = elmoModel.get_elmo_scores(query_string, elmoVectorsForFundObjectives)
        print(elmoScores)
        top_scores["ELMO"] = [round(x[1],4) for x in elmoScores]
        elmoIdx = [x[0] for x in elmoScores]
        top_objectives["ELMO"]=[fund_objectives[int(idx)] for idx in elmoIdx]
        top_docIdx["ELMO"] = [int(idx) for idx in elmoIdx]
        #####################USE - Universal Sentence Encoder
        # USE Model
        #elmoScores = elmoModel.get_elmo_scores(query_string, elmoVectorsForFundObjectives, elmoModelPreTrained, graph, session)
        print('===============USEModel ==============', query_string)
        useScores = useModel.get_use_scores(query_string, useVectorsForFundObjectives)
        print(useScores)
        top_scores["USE"] = [round(x[1],4) for x in useScores]
        useIdx = [x[0] for x in useScores]
        top_objectives["USE"]=[fund_objectives[int(idx)] for idx in useIdx]
        top_docIdx["USE"] = [int(idx) for idx in useIdx]
        #####################LSI
        # LSI Model
        print('===============LSIModel ==============', query_string)
        lsiScores = lsiModel.get_lsi_scores(query_string, dictionary, lsi_model, lsi_doc_index)
        print('LSI scores:', lsiScores)
        top_scores["LSI"] = [round(x[1],4) for x in lsiScores]
        lsiIdx = [x[0] for x in lsiScores]
        top_objectives["LSI"]=[fund_objectives[int(idx)] for idx in lsiIdx]
        top_docIdx["LSI"] = [int(idx) for idx in lsiIdx]
        #####################TFIDF
        # TFIDF Model
        print('===============TFIDFModel ==============', query_string)
        tfidfScores = tfIdfModel.get_tfidf_scores(query_string, dictionary, tfidf_model, tfidf_doc_index)
        print('TFIDF scores:', tfidfScores)
        top_scores["TFIDF"] = [round(x[1],4) for x in tfidfScores]
        tfidfIdx = [x[0] for x in tfidfScores]
        top_objectives["TFIDF"]=[fund_objectives[int(idx)] for idx in tfidfIdx]
        top_docIdx["TFIDF"] = [int(idx) for idx in tfidfIdx]
        #####################BERT
        # BERT Model
        #bertScores = bertModel.get_bert_scores(query_string, bertClient, bert_vec)
        #print('BERT scores:', bertScores)
        #top_scores["BERT"] = [round(x[1],4) for x in bertScores]
        #bertIdx = [x[0] for x in bertScores]
        #top_objectives["BERT"]=[fund_objectives[int(idx)] for idx in bertIdx]
        #top_docIdx["BERT"] = [int(idx) for idx in bertIdx]
        #####################FastText
        # fastTest Model
        print('===============FastTextModel ==============', query_string)
        fastTextScores = fastTextModel.get_fastText_scores(query_string, ft_model_load, data_cleaned)
        print('fastText scores:', fastTextScores)
        top_scores["FastText"] = [round(x[1],4) for x in fastTextScores]
        fastTextIdx = [x[0] for x in fastTextScores]
        top_objectives["FastText"]=[fund_objectives[int(idx)] for idx in fastTextIdx]
        top_docIdx["FastText"] = [int(idx) for idx in fastTextIdx]
    return render_template('model.html', topScores=top_scores, message=query_string, topObjectives=top_objectives, topDocIndex = top_docIdx)

################################
# contact page routing
################################
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

################################
# helper functions to load and pre-process
################################

def preProcessData(fund_objectives):
    #df = pd.read_csv('fo.csv', encoding='ISO-8859-1')
    #sentences = df['Objective'].values
    data_cleaned = preProcess.pre_process_sent(fund_objectives)
    return data_cleaned

def loadObjectives():
    df = pd.read_csv('fo.csv', encoding='ISO-8859-1')
    sentences = df['Objective'].values
    return sentences

################################
# Load the data and pre-process it
################################
fund_objectives = loadObjectives()
data_cleaned = preProcessData(fund_objectives)

################################
# Train the word2VecModel
################################
w2vModel = word2VecModel.trainWord2VecModel(data_cleaned)

################################
# Load pre-trained elmo vectors for the fund objectives
################################
pickle_in = open("elmo_vectors_03122019.pickle", "rb")
elmoVectorsForFundObjectives = pickle.load(pickle_in)



################################
# Load pre-trained USE vectors for the fund objectives
################################
pickle_use = open("USE_vectors_03122019.pickle", "rb")
useVectorsForFundObjectives = pickle.load(pickle_use)

################################
# LSI Model - using gemsim
################################

dictionary = corpora.Dictionary(data_cleaned)
bow_corpus = [dictionary.doc2bow(text) for text in data_cleaned]
# train the model
tfidf = models.TfidfModel(bow_corpus)
tfidf_corpus = tfidf[bow_corpus]
lsi_model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)
lsi_doc_index = similarities.MatrixSimilarity(lsi_model[bow_corpus])  # transform corpus to LSI space and index it

################################
# TFIDF Model - using gemsim
################################

#dictionary = corpora.Dictionary(data_cleaned)
#bow_corpus = [dictionary.doc2bow(text) for text in data_cleaned]
# train the model
tfidf_model = models.TfidfModel(bow_corpus)
#tfidf_corpus = tfidf_model[bow_corpus]
tfidf_doc_index = similarities.MatrixSimilarity(tfidf_model[bow_corpus])  # transform corpus to TFIDF space and index it

################################
# BERT Model -
################################
#
#bertClient = BertClient()
#data_cleaned

#bert_vec=bertClient.encode(data_cleaned,is_tokenized=True)

################################
# FastText Model -
################################

ft_model_load = FastText.load("fast_model_embeddings_20191207.model")
#ft_model_load = FastText(data_cleaned, size = 60, window=40,min_count=5, sample=0.01, sg=1, iter=100)
################################
# __main__
################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80,debug=True)
