#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 11:56:32 2021

@author: asrafatima
"""

import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import networkx as nx
import pickle
import pandas as pd
import re
import numpy as np
from collections import Counter
from scipy.stats import entropy
from nltk.corpus import wordnet
from keras import backend as K
from tensorflow import keras
import os
import warnings
import sys

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

nlp2 = spacy.load('en_core_web_md')
stopword = stopwords.words('english')
stopword.extend(['take', 'call', 'get', 'first', 'second', 'third', 'need', 'needs', 'dont', 'want', 'give', 'go', 'given', 'turns', 'way', 'i\'m', 'seems', 'place',
                 'one', 'find', 'told', 'feeling', 'year', 'years', 'told', 'say', 'went', 'got', 'thing', 'take', 'taking', '\'m', '\'re', 'become', 'became', 'made', 'life'])


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def outer(cell):
    def inner(cell):
        return WordNetLemmatizer().lemmatize(cell, pos='v')
    return inner(cell)


with open('deployables/lemmedWords.pkl', 'rb') as f:
    lemmedWords = pickle.load(f)
antonyms = pd.read_pickle("deployables/antonyms.pkl")
G = nx.read_gpickle("deployables/graph.pkl")
flatDF_weight = pd.read_pickle("deployables/flatDF.pkl")

lemmedIndex = dict()
for i, v in enumerate(lemmedWords):
    lemmedIndex[v] = i


depressionModel = keras.models.load_model(
    'deployables/depression.h5', custom_objects={"coeff_determination": coeff_determination})
anxietyModel = keras.models.load_model(
    'deployables/anxiety.h5', custom_objects={"coeff_determination": coeff_determination})
stressModel = keras.models.load_model(
    'deployables/stress.h5', custom_objects={"coeff_determination": coeff_determination})


def distance(row):
    Emotsum = 0
    happyEmot = 0
    sadEmot = 0
    depressionEmot = 0
    anxietyEmot = 0
    stressEmot = 0
    fearEmot = 0
    dist = []
    for i, v in enumerate(row.iloc[1:], 1):

        x = row.iloc[i-1]
        y = v
        if x not in G.nodes() or y not in G.nodes():
            continue

        Emotsum += nx.shortest_path_length(G, x, y)
        dist.append(nx.shortest_path_length(G, x, y))
        if i == 1:
            happyEmot += nx.shortest_path_length(G, x, 'happy')
            sadEmot += nx.shortest_path_length(G, x, 'sad')
            depressionEmot += nx.shortest_path_length(G, x, 'depression')
            anxietyEmot += nx.shortest_path_length(G, x, 'anxiety')
            stressEmot += nx.shortest_path_length(G, x, 'stress')
            fearEmot += nx.shortest_path_length(G, x, 'fear')
        happyEmot += nx.shortest_path_length(G, y, 'happy')
        sadEmot += nx.shortest_path_length(G, y, 'sad')
        depressionEmot += nx.shortest_path_length(G, x, 'depression')
        anxietyEmot += nx.shortest_path_length(G, x, 'anxiety')
        stressEmot += nx.shortest_path_length(G, x, 'stress')
        # fearEmot+=nx.shortest_path_length(G,x,'fear')
    counter = Counter(dist)
    probs = []
    for i, v in counter.most_common():
        probs.append(v/len(dist))
    return Emotsum, happyEmot, sadEmot, depressionEmot, anxietyEmot, stressEmot, entropy(probs)


def ret_results(text):

    refinednote = re.split('[\n.]', text)
    selectedWords = set()
    vector_representation = np.zeros((len(lemmedWords)), dtype=float)
    for sent in refinednote:
        note = nlp2(sent)

        isNeg = False

        for token in note:

            if token.dep_ == 'neg':
                isNeg = not isNeg
                count = 0
            if token.text.lower() in stopword or outer(token.text.lower()) in stopword:
                continue

            if token.pos_ in ['VERB', 'ADJ', 'NOUN']:

                if isNeg:
                    if count >= 2:
                        count = 0
                        isNeg = not isNeg
                    else:
                        count += 1

                if isNeg:

                    sims = np.array([nlp2(outer(token.text.lower())).similarity(
                        nlp2(lemmedWord)) for lemmedWord in lemmedWords])

                    word = antonyms[antonyms[0] == outer(
                        token.text.lower())][1].values

                    if len(word) == 0:
                        if len(antonyms[antonyms[0] == lemmedWords[sims.argmax()]][1].values) == 0:
                            continue
                        else:

                            opword = antonyms[antonyms[0] ==
                                              lemmedWords[sims.argmax()]][1].values[0]
                            sims = np.array([nlp2(opword).similarity(
                                nlp2(lemmedWord)) for lemmedWord in lemmedWords])

                            selectedWords.add(lemmedWords[sims.argmax()])
                            if sims.max() == 1.0:

                                vector_representation[sims.argmax()] = 0.999
                            else:
                                vector_representation[sims.argmax(
                                )] = sims.max()
                            isNeg = False
                            continue
                    else:
                        sims = np.array([nlp2(word[0]).similarity(
                            nlp2(lemmedWord)) for lemmedWord in lemmedWords])

                        selectedWords.add(lemmedWords[sims.argmax()])
                        if sims.max() == 1.0:

                            vector_representation[sims.argmax()] = 0.999
                        else:
                            vector_representation[sims.argmax()] = sims.max()
                        isNeg = False
                        continue
                if outer(token.text.lower()) in lemmedWords:

                    vector_representation[lemmedWords.index(
                        outer(token.text.lower()))] = 0.99
                    selectedWords.add(token.text.lower())
                    continue
                if outer(token.text.lower()) in G.nodes():

                    neibs = [i for i in nx.neighbors(
                        G, outer(token.text.lower())) if i in lemmedWords]

                    if len(neibs) == 0:
                        continue
                    sims = np.array(
                        [nlp2(outer(token.text.lower())).similarity(nlp2(neib)) for neib in neibs])
                    if sims.max() < 0.5:

                        continue
                    selectedWords.add(neibs[sims.argmax()])

                    if sims.max() == 1.0:

                        vector_representation[lemmedIndex[neibs[sims.argmax()]]
                                              ] = 0.999
                    else:
                        vector_representation[lemmedIndex[neibs[sims.argmax()]]] = sims.max(
                        )
                else:

                    sims = np.array([nlp2(outer(token.text.lower())).similarity(
                        nlp2(lemmedWord)) for lemmedWord in lemmedWords])
                    if sims.max() < 0.5:
                        continue
                    selectedWords.add(lemmedWords[sims.argmax()])

                    if sims.max() == 1.0:

                        vector_representation[sims.argmax()] = 0.999
                    else:
                        vector_representation[sims.argmax()] = sims.max()

    stress, anxiety, depression = 0, 0, 0
    testSample = []

    if len(selectedWords) != 0:

        testWords = np.array(list(selectedWords))
        testWords = testWords.reshape((1, -1))
        test = pd.DataFrame(testWords)
        test['EmotionSum'] = test.apply(lambda row: distance(row), axis=1)
        test[['EmotionSum', 'happyEmotions', 'SadEmotions', 'depressedEmotions', 'anxietyEmotions',
              'stressEmotions', 'entropy']] = pd.DataFrame(test.EmotionSum.to_list(), index=test.index)
        test = test.iloc[:, -7:]
        testSample = pd.concat([pd.DataFrame(vector_representation.reshape(
            1, -1), columns=flatDF_weight.columns), test], axis=1)

        depression = depressionModel.predict(testSample.values)[0][0]
        anxiety = anxietyModel.predict(testSample.values)[0][0]
        stress = stressModel.predict(testSample.values)[0][0]

    return depression, anxiety, stress


while(True):
    text = input("Enter text you want  to analyze or enter exit to terminate")
    if text.lower() == 'exit':
        sys.exit()

    d, a, s = ret_results(text)
    print(f'depression={d}')
    print(f'anxiety={a}')
    print(f'stress={s}')
