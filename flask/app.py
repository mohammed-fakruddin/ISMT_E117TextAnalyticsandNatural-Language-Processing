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


#=================
# custom python modules
#=================
from pythonModules import preProcess
from nlpModels.word2VecModel import word2VecModel
from nlpModels.elmoModel import elmoModel
from nlpModels.useModel import useModel
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
    top_scores={"Cosine":[], "Word2Vec":[], "USE":[], "Glove":[], "BERT":[], "ELMO":[], "Jaccard":[]}
    top_objectives={"Cosine":[], "Word2Vec":[], "USE":[], "Glove":[], "BERT":[], "ELMO":[], "Jaccard":[]}
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
        w2vScores = word2VecModel.get_word2Vec_scores(query_string, w2vModel)
        top_scores["Word2Vec"] = [round(x[1],4) for x in w2vScores]
        w2vDocIdx = [x[0] for x in w2vScores]
        top_objectives["Word2Vec"]=[fund_objectives[int(idx)] for idx in w2vDocIdx]
        #####################
        # ELMO Model
        #elmoScores = elmoModel.get_elmo_scores(query_string, elmoVectorsForFundObjectives, elmoModelPreTrained, graph, session)
        elmoScores = elmoModel.get_elmo_scores(query_string, elmoVectorsForFundObjectives)
        print(elmoScores)
        top_scores["ELMO"] = [round(x[1],4) for x in elmoScores]
        elmoIdx = [x[0] for x in elmoScores]
        top_objectives["ELMO"]=[fund_objectives[int(idx)] for idx in elmoIdx]
        #####################USE - Universal Sentence Encoder
        # USE Model
        #elmoScores = elmoModel.get_elmo_scores(query_string, elmoVectorsForFundObjectives, elmoModelPreTrained, graph, session)
        useScores = useModel.get_use_scores(query_string, useVectorsForFundObjectives)
        print(useScores)
        top_scores["USE"] = [round(x[1],4) for x in useScores]
        useIdx = [x[0] for x in useScores]
        top_objectives["USE"]=[fund_objectives[int(idx)] for idx in useIdx]


        print(sims)

        #print(request.form['message'])
    return render_template('model.html', topScores=top_scores, message=query_string, topObjectives=top_objectives)

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
# __main__
################################

if __name__ == '__main__':
    app.run(debug=True)
