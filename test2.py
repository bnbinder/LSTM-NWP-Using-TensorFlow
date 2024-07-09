#import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

#read dataset
with open("tt.txt") as file:
    text = file.read()

    #Tokenizer process
tokenizer = Tokenizer()
#fit
tokenizer.fit_on_texts([text])
#assign length of word index
total_words = len(tokenizer.word_index) + 1

#declare ngrams
input_sequences = []
#split the sentence from '\n'
for line in text.split('\n'):
    #get tokens
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

setence_token = input_sequences[3] # [1, 1561, 5, 129, 34]
sentence = []
for token in setence_token:
    sentence.append(list((tokenizer.word_index).keys())[list((tokenizer.word_index).values()).index(token)])
print(sentence)

#maximum sentence length
max_sequence_len = max([len(seq) for seq in input_sequences])
# input sequences
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

#convert one-hot-encode
y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))

#create model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
print(model.summary())

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#fit the model
model.fit(X, y, epochs=100, verbose=1)

#determine a text
seed_text = "I will close the door if"
# predict word number
next_words = 7

for _ in range(next_words):
    #convert to token
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    #path sequences
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    #model prediction
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    # get predict words
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)