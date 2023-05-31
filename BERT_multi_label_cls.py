import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa

data = pd.read_csv("/kaggle/input/best-books-10k-multi-genre-data/goodreads_data.csv").drop("Unnamed: 0", axis=1)

# Preprocess the data
data = data.dropna()
data = data.drop_duplicates()

#Extract all the distinct genres from the data
genre_set = set()
for s in data['Genres']:
    # Extracting genres from the string
    genres = [genre.strip(" '") for genre in s.strip("[]").split(",")]

    # Creating a set of genres
    genres = set(genres)
    genre_set.update(genres)

genre_set.remove('')

print(len(genre_set)) #gives 617

#Getting count of each genre in the data
genre_list = list()
for s in data['Genres']:
    # Extracting genres from the string
    genres = [genre.strip(" '") for genre in s.strip("[]").split(",")]

    genre_list.extend(genres)
    
from collections import Counter
genres_count = dict(Counter(genre_list))

genres_count_sorted = {k: v for k, v in sorted(genres_count.items(), key=lambda item: item[1],
                                              reverse=True)}
import statistics
counts = list(genres_count.values())
print(max(counts))
print(min(counts))
print(sum(counts)/len(counts))
print(statistics.median(counts))

#average frequency of each genre is around 95
top_genres = {k:v for k, v in genres_count.items() if v>=95}
print(len(top_genres)) #gives 100

top_genres = list(genres_count_sorted.keys())[:10] #taking top 10 genres for avoiding sparsity problem during model training

##Plotting frequencies of top 10 genres

import matplotlib.pyplot as plt
# Extracting the top 10 genres and their corresponding counts
genres = list(genres_count_sorted.keys())[:10]
counts = list(genres_count_sorted.values())[:10]
# Creating a bar plot
plt.bar(genres, counts)
# Adding labels and title
plt.xlabel('Genres')
plt.ylabel('Count')
plt.title('Genre Count')
# Rotating the x-axis labels for better readability (optional)
plt.xticks(rotation=45)
# Displaying the plot
plt.show()

##Check length of genres list for each row

genres_length = set()
for s in data['Genres']:
    # Extracting genres from the string
    genres = [genre.strip(" '") for genre in s.strip("[]").split(",")]

    # Creating a set of genres
    length = len(genres)
    genres_length.add(length)

print(genres_length) #gives {1,2,3,4,5,6,7}

#List of genres - to be used by multilabelbinarizer
genre_list = list()

for s in data['Genres']:
    
    genres = [genre.strip(" '") for genre in s.strip("[]").split(",")]
    
    genres = [x for x in genres if x in list(top_genres.keys())]
    
    genre_list.append(genres) #it is a list of lists
    
mlb = MultiLabelBinarizer()

genre_labels = mlb.fit_transform(genre_list)

preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

y = genre_labels

x_train, x_test, y_train, y_test = train_test_split(data['Description'], y, test_size=0.30)

num_classes = 100

##Now, define the model

k = 7
i = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
x = preprocessor(i)
x = encoder(x)
x = tf.keras.layers.Dropout(0.2, name="dropout")(x['pooled_output'])
x = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)
x = tf.keras.layers.Softmax(name="probabilities")(x)

model = tf.keras.Model(i, x)

##Define all metrics and params and train the model
n_epochs = 20


earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", 
                                                      patience = 3,
                                                      restore_best_weights = True)

f1score = tfa.metrics.F1Score(num_classes, average='weighted')

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[f1score])

model_fit = model.fit(x_train, 
                      y_train, 
                      batch_size=32,
                      epochs = n_epochs,
                      validation_data = (x_test, y_test),
                      callbacks = [earlystop_callback])

#The F1 val score was around 0.40

def predict_genres(test_sample, k):
    '''function to predict genres for a given book description'''
    
    input_data = np.expand_dims(test_sample, axis=0)  # Reshape the input data

    predictions = model.predict(input_data)

    # Getting the indices of the top 7 probabilities
    top_indices = np.argsort(predictions)[0, -k:][::-1]
    
    top_classes = mlb.classes_[top_indices]
    
    return top_classes
  
predict_genres(x_test.iloc[2], 7) #['Fiction', 'Fantasy', 'Adventure', 'Epic Fantasy', 'Science Fiction', 'Science Fiction Fantasy', 'Romance']

#Compare with actual
data['Genres'].iloc[2] #"['Classics', 'Fiction', 'Romance', 'Historical Fiction', 'Literature', 'Historical', 'Audiobook']"






        


    
    
    
