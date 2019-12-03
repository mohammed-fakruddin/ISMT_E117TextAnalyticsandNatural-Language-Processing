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

stop_words = set(stopwords.words('english'))
custom_stop_words=['fund', 'funds', 'seek', 'seeks']
stop_words.update(custom_stop_words)



def pre_process_sent(sentences):
    docs=[]
    if isinstance(sentences, str) :
        sentences = [[sentences]]
    #print(sentences)
    for s in sentences:
        #remove punctuation
        s = str(s).translate(str.maketrans('','', string.punctuation))
        #lower
        s = s.lower()
        temp=[]
        #split sentences into words
        for w in word_tokenize(s):
            #remove stops
            if w not in stop_words:
                #w = ps.stem(w)
                #w = lemmatizer.lemmatize(w)
                temp.append(w)
        docs.append(temp)
    return docs

def pre_process_sent_with_stemmer(sentences):
    docs=[]
    if isinstance(sentences, str) :
        sentences = [[sentences]]
    for s in sentences:
        #remove punctuation
        s = str(s).translate(str.maketrans('','', string.punctuation))
        #lower
        s = s.lower()
        temp=[]
        #split sentences into words
        for w in word_tokenize(s):
            #remove stops
            if w not in stop_words:
                w = porter.stem(w)
                #w = lemmatizer.lemmatize(w)
                temp.append(w)
        docs.append(temp)
    return docs

def pre_process_sent_with_lemma(sentences):
    docs=[]
    if isinstance(sentences, str) :
        sentences = [[sentences]]
    for s in sentences:
        #remove punctuation
        s = str(s).translate(str.maketrans('','', string.punctuation))
        #lower
        s = s.lower()
        temp=[]
        #split sentences into words
        for w in word_tokenize(s):
            #remove stops
            if w not in stop_words:
                w = lemmatizer.lemmatize(w)
                temp.append(w)
        docs.append(temp)
    return docs

def get_just_tokens(sentences):
    tokens=[]
    if isinstance(sentences, str) :
        sentences = [sentences]
    for s in sentences:
        #for w in s:
        tokens.append(s.split())
    return tokens

def get_vocabulary_for_docs(doc_tokens_cleaned):
    vocab = []
    if isinstance(doc_tokens_cleaned, str) :
        doc_tokens_cleaned = [[doc_tokens_cleaned]]
    for sent_word in doc_tokens_cleaned:
        for w in sent_word:
            vocab.append(w)
    return vocab

def get_total_word_count(sentences_tokens):
    total = 0
    for s in sentences_tokens:
        for w in s:
            total = total + len(w.split())
    return total

def get_min_max_sentence_length(sentences):
    minimum = 9999999
    maximum = 0
    for s in sentences:
        if (len(s) > maximum):
            maximum = len(s)
        if (len(s) < minimum):
            minimum = len(s)
    return (minimum, maximum)
