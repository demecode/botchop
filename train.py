import random

import nltk

# punkt is a tokenizer divides a text into a list of sentences
# by using an unsupervised algorithm to build a model for abbreviation
# words, collocations, and words that start sentences
nltk.download('punkt')
nltk.download('wordnet')  ## corpus word reader
from nltk.stem import WordNetLemmatizer  # Returns the input word unchanged if it cannot be found in WordNet.

lemmatizer = WordNetLemmatizer()
import json
import pickle  ## serialization of data

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data_objects.json').read()
data_objects = json.loads(data_file)

for data_ob in data_objects['intents']:
    for pattern in data_ob['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding a document with the data
        documents.append((w, data_ob['tag']))

        # add classes to the class list
        if data_ob['tag'] not in classes:
            classes.append(data_ob['tag'])

# our json file has subjects within objects - so I'll need to create a nested for loop to access
# and extract all words and add them to the words list
# first we lematize each word and bring the word back to its base which is easier to train the model
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# print(words)

# sorted the classes list from the data
classes = sorted(list(set(classes)))
# print(classes)
#
# print(len(documents), "documents",documents)
# print(len(classes), "classes", classes)
# print(len(words), "unique lematized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# build the model

train = []
output_empty = [0] * len(classes)  # 5 classes so this create an empty array of five 0s (columns)
print(output_empty, "CRAY")
for doc in documents:
    bag = []  # initialize bag as empty words
    pattern_words = doc[0]  # list of tokenized words for the pattern - e.g get the patterns words for each object
    # print (pattern_words, "PATTERN_WORDS")
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in
                     pattern_words]  # lematize each word - make the words make sense
    # print(pattern_words, 'WOAH')
    # create a bag of words array with 1, if a word match found in the current pattern
    for w in words:
        # print(words, "OVER and OVER")
        # print(w, "THIS IS THE MADEEEEEEEEEE")
        # so for each letter found in the patter words, we add 1 to the bag else we add a 0
        bag.append(1) if w in pattern_words else bag.append(0)
        # print(bag)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1  # output is a '0' for each tag and '1' for current tag (for each pattern)
        print(output_row, 'NEVER LET ME DOWN')
        train.append([bag, output_row])

# shuffle the features and turn into a np.array
random.shuffle(train)
train = np.array(train, dtype=object)
print(train)

# now we want to create test lists using X - patterns, y - intents
#
train_x = list(train[:, 0])
train_y = list(train[:, 1])
print(train_x, "X DATA")
print(train_y, "Y DATA")

# now to create the model

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
final = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('bot_model.h5', final)

print("model created")
