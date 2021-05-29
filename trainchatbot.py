#Libraries I'm going to use
import random
import json
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words=[]
classes=[]
documents=[]
ignore_letters = ['!', '?', ',', '.']


#Accessing the intents keys from the JSON file
#Tokenize in order to categorize certain words in specific words and ignore the rest

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#ignore duplicates
words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes=sorted(set(classes))


#save duplicates in pickle file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))



#train nn, asign numerical values to words so them are trained in that particular pattern

training = []
oputput_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns =document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(oputput_empty)
    output_row [classes.index(document[1])] = 1
    training.append([bag, output_row])

#shuffle the data
random.shuffle(training)
training = np.array(training)

#x & y values, the features and values that are going to train the nn
train_x = list(training[:, 0])
train_y = list(training[:, 1])

#create the model

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#define a stochastic grade descendence

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print('Done')
