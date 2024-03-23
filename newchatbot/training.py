import json
import random
import nltk
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# use to load intents from json file
try:
    with open('intents.json', 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' file not found.")
    exit()
except json.JSONDecodeError:
    print("Error: Unable to parse 'intents.json'. Check if it's properly formatted.")
    exit()

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Iterate through intents and extract patterns and responses
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each pattern into words
        word_list = nltk.word_tokenize(pattern)

        words.extend([lemmatizer.lemmatize(word.lower()) for word in word_list if word not in ignore_letters])
       
        documents.append((word_list, intent['tag']))
   
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Perform additional preprocessing
words = sorted(set(words))
classes = sorted(set(classes))


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    combined_feature_label = bag + output_row
    
    training.append(combined_feature_label)

# Shuffle training data
random.shuffle(training)


training = np.array(training)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Build and compile the model so as to work with the chatbot.py
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# trainigng rate to increase accuracy
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#to tvain the model
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# to save the model
model.save('chatbotmodel.h5')
print("Model trained and saved successfully.")
