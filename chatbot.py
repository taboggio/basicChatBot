#building the chatbot
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents =json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
model=load_model('chatbotmodel.h5')


#creating the functions

def clean_up_sentence(sentence):
    sentence_words =nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]

#converte sentence in bag of words of 1 and 0 to be trained in nn
def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag= [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

#prediction function with error_thr of 0.25
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r]for i,r in enumerate(res)if r>ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1], reverse=True)
    returnlist=[]
    for r in results:
        returnlist.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return returnlist
