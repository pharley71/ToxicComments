import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

EMBEDDING_FILE=f'glove6b50d/glove.6B.50d.txt'

def load_data(filename):
    columns = ['comment_text', 'toxic']
    return pd.read_csv(filename, usecols = columns)

def class_metrics(y_test, y_preds):
    print('Accuracy score: ', format(accuracy_score(y_test,y_preds)))
    print('Precision score: ', format(precision_score(y_test, y_preds)))
    print('Recall score: ', format(recall_score(y_test, y_preds)))
    print('F1 score: ', format(f1_score(y_test,y_preds)))
    print("Confusion Matrix: ") 
    print(confusion_matrix(y_test, y_preds))
  
print("loading data ...")    
columns = ['comment_text', 'toxic']
data = pd.read_csv('train.csv', usecols = columns)
X = data['comment_text']
y = data['toxic']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)
    
print("tokenizing ... ")
# Tokenize the text comments 
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
list_tokenized_train = tokenizer.texts_to_sequences(X_train)
list_tokenized_test = tokenizer.texts_to_sequences(X_test)

print("padding ...")
#Add padding
maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

#Build model
embed_size = 128
inp = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(10, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()

file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=4)

callbacks_list = [checkpoint, early]

batch_size = 64
epochs = 2
model.fit(X_t, y_train, 
          batch_size=batch_size, 
          epochs=epochs,
          callbacks = callbacks_list,           
          #validation_data = (X_te, y_test))
          validation_split=0.1, 
          verbose = 2) 
preds = model.predict(X_te) 
#class_preds = (preds > 0.5).astype("int32") 
#print("First 1000 predictions:", preds) 
preds[preds < 0.5] = 0
#print("20 predictions refactored:", preds[:20])
preds[preds >= 0.5] = 1
preds = preds.astype("int32")
print("Number of positives: ", np.sum(preds))
class_metrics(y_test, preds)
         
          