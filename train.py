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
from keras.models import Sequential
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
print(words)

# sorted the classes list from the data
classes = sorted(list(set(classes)))
print(classes)

print(len(documents), "documents")
print(len(classes), "classes")
print(len(words), "unique lematized words")

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# build the model

train = []
output_empty = [0] * len(classes)
print(output_empty, "CRAY")
for doc in documents:
    bag = [] # initialize bag as empty words
    pattern_words = doc[0] # list of tokenized words for the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words] # lematize each word - make the words make sense
    print(pattern_words, 'WOAH')
    # create a bag of words array with 1, if a word match found in the current pattern
    for w in words:
        print(w, "THIS IS THE MADEEEEEEEEEE")
        bag.append(1) if w in pattern_words else bag.append(0)
        print(bag)

        output_row = list(output_empty)
        print(output_row)
        output_row[classes.index(doc[1])] = 1
        print(output_row)

